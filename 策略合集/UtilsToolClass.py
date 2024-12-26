from kuanke.user_space_api import *
from jqdata import *
from jqfactor import get_factor_values
import datetime
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib
from datetime import date as dt
import math
import talib as tl
from jqlib.technical_analysis import *
from scipy.linalg import inv
import pickle
import requests
import datetime as datet
from prettytable import PrettyTable
import inspect


class UtilsToolClass:
    def __init__(self):
        self.name = None
        self.subportfolio_index = None

    def set_params(self, name, subportfolio_index):
        self.name = name
        self.subportfolio_index = subportfolio_index

    ##################################  交易函数群 ##################################

    # 开仓单只
    def open_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        now_time = context.current_dt.time()
        current_data = get_current_data()
        before_buy = datetime.time(9, 30) > now_time
        # log.info('before_buy:',before_buy)
        style_arg = MarketOrderStyle(current_data[security].day_open) if before_buy else None
        if target:
            order_info = order_target_value(security, value, style=style_arg, pindex=self.subportfolio_index)
        else:
            # log.info('S:', security, "--value:", value)
            order_info = order_value(security, value, style=style_arg, pindex=self.subportfolio_index)

        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
        if order_info != None and order_info.filled > 0:
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--买入股票: {security} "
                       f"--计划买入金额: {value} "
                       f"--买入数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--买入均价: {order_info.price} "
                       f"--实际买入金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f}\n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--买入股票，交易失败！！股票: {security} "
                   f"--失败原因: {order_info} "
                   f"--计划买入金额: {value}\n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    # 清仓单只
    def close_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        if target:
            order_info = order_target_value(security, value, pindex=self.subportfolio_index)
        else:
            order_info = order_value(security, value, pindex=self.subportfolio_index)
        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"

        if order_info != None and order_info.status == OrderStatus.held and order_info.filled == order_info.amount:
            # 计算收益率:（当前价格/持仓价格）- 1
            ret = 100 * (order_info.price / order_info.avg_cost - 1)
            # 计算收益金额: 可卖仓位 *（当前价格/持仓价格)
            ret_money = order_info.amount * (order_info.price - order_info.avg_cost)
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--卖出股票: {security} "
                       f"--卖出数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--持仓均价: {order_info.avg_cost} "
                       f"--卖出均价: {order_info.price} "
                       f"--实际卖出金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f} 收益率: {ret:.2f}% 收益金额: {ret_money:.2f} \n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--失败原因: {order_info} "
                   f"--卖出股票，交易失败！！！股票: {security} \n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    ##################################  选股函数群 ##################################

    # 获取股票股票池（暂无使用）
    def get_security_universe(self, context, security_universe_index, security_universe_user_securities):
        log.info(self.name, '--get_security_universe函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        temp_index = []
        for s in security_universe_index:
            if s == 'all_a_securities':
                temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
            else:
                temp_index += get_index_stocks(s)
        for x in security_universe_user_securities:
            temp_index += x
        return sorted(list(set(temp_index)))

    # 过滤科创北交
    def filter_kcbj_stock(self, context, stock_list):
        log.info(self.name, '--filter_kcbj_stock过滤科创北交函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        for stock in stock_list[:]:
            if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[:2] == '30':
                stock_list.remove(stock)
        return stock_list

    # 过滤停牌股票
    def filter_paused_stock(self, context, stock_list):
        log.info(self.name, '--filter_paused_stock过滤停牌股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    # 过滤ST及其他具有退市标签的股票
    def filter_st_stock(self, context, stock_list):
        log.info(self.name, '--filter_st_stock过滤ST及其他具有退市标签的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        return [stock for stock in stock_list
                if not current_data[stock].is_st
                and 'ST' not in current_data[stock].name
                and '*' not in current_data[stock].name
                and '退' not in current_data[stock].name]

    # 过滤涨停的股票
    def filter_highlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_highlimit_stock过滤涨停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] < current_data[stock].high_limit]

    # 过滤跌停的股票
    def filter_lowlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_lowlimit_stock过滤跌停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] > current_data[stock].low_limit]

    # 过滤次新股（小市值专用）
    def filter_new_stock(self, context, stock_list, days):
        log.info(self.name, '--filter_new_stock过滤次新股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        return [stock for stock in stock_list if
                not context.previous_date - get_security_info(stock).start_date < datetime.timedelta(days=days)]

    # 过滤大幅解禁（小市值专用）
    def filter_locked_shares(self, context, stock_list, days):
        log.info(self.name, '--filter_locked_shares过滤解禁股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取指定日期区间内的限售解禁数据
        df = get_locked_shares(stock_list=stock_list, start_date=context.previous_date.strftime('%Y-%m-%d'),
                               forward_count=days)
        # 过滤出解禁数量占总股本的百分比超过 20% 的股票
        df = df[df['rate1'] > 0.2]
        filterlist = list(df['code'])
        # 从股票池中排除这些股票
        return [stock for stock in stock_list if stock not in filterlist]

    ###################################  公用函数群 ##################################
    # 获取个股行业,暂无使用
    def get_industry_name(self, i_Constituent_Stocks, value):
        return [k for k, v in i_Constituent_Stocks.items() if value in v]

    # 把prettytable对象转换成键值对字符串
    def pretty_table_to_kv_string(self, table):
        headers = table.field_names
        result = ""
        data_rows = table._rows  # 直接获取表格内部存储的数据行列表，避免格式干扰
        for row in data_rows:
            for header, cell in zip(headers, row):
                result += f"{header}: {cell}\n<br>"
            result += "\n<br>"
        return result.rstrip()

    # 发送微信消息
    def send_wx_message(self, context, item, message):
        if context.is_send_wx_message != 1:
            return
        url = "https://wxpusher.zjiecode.com/api/send/message"

        data = {
            "appToken": "AT_B7CVGazuAWXoqBoIlGAzlIwkunQuXIQM",
            "content": f"<h1>{item}</h1><br/><p style=\"color:red;\">{message}</p>",
            "summary": item,
            "contentType": 2,
            "topicIds": [
                36105
            ],
            "url": "https://wxpusher.zjiecode.com",
            "verifyPay": False,
            "verifyPayType": 0
        }
        response = requests.post(url, json=data)
        # 可以根据需要查看响应的状态码、内容等信息
        # print(response.status_code)
        # print(response.text)

    def Market_temperature(self, context, market_temperature='warm'):
        log.info(self.name, '--Market_temperature函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 获取数据：使用attribute_history函数获取沪深300指数过去220天的收盘价数据。
        index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']

        # 计算市场高度：通过计算最近5天收盘价的平均值与过去220天收盘价的最小值之差，再除以过去220天收盘价的最大值与最小值之差，得到市场高度（market_height)。
        market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))

        # 判断市场温度：根据市场高度的值，将市场温度分为三种状态：
        # ·如果市场高度小于0.20, 则市场温度为"cold"。
        # ·如果市场高度大于0.90, 则市场温度为"hot"。
        # ·如果过去60天内的最高收盘价与最低收盘价之比大于1.20, 则市场温度为"warm"
        if market_height < 0.20:
            market_temperature = "cold"
        elif market_height > 0.90:
            market_temperature = "hot"
        elif max(index300[-60:]) / min(index300) > 1.20:
            market_temperature = "warm"

        return market_temperature

    # 4-1 打印每日持仓信息,暂无使用
    def print_position_info(self, context):
        log.info(self.name, '--print_position_info函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 打印当天成交记录
        trades = get_trades()
        for _trade in trades.values():
            log.info('成交记录：' + str(_trade))
        # 打印账户信息
        for position in list(context.portfolio.positions.values()):
            securities = position.security
            cost = position.avg_cost
            price = position.price
            ret = 100 * (price / cost - 1)
            value = position.value
            amount = position.total_amount
            log.info('代码:{}'.format(securities))
            log.info('成本价:{}'.format(format(cost, '.2f')))
            log.info('现价:{}'.format(price))
            log.info('收益率:{}%'.format(format(ret, '.2f')))
            log.info('持仓(股):{}'.format(amount))
            log.info('市值:{}'.format(format(value, '.2f')))
            log.info('———————————————————————————————————')
        log.info('———————————————————————————————————————分割线————————————————————————————————————————')

    # 筛选出某一日涨停的股票
    def get_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['close', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停的股票，含炸板的
    def get_ever_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停但未封板的股票
    def get_ever_hl_stock2(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily',
                        fields=['close', 'high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close!=high_limit and high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()
