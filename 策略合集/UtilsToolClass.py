from kuanke.user_space_api import *
from jqdata import *
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib
import math
import talib as tl
from jqlib.technical_analysis import *
from scipy.linalg import inv
import pickle
import requests
import datetime as dt
from prettytable import PrettyTable
import inspect


class UtilsToolClass:
    def __init__(self):
        self.name = None
        self.subportfolio_index = None

    def set_params(self, name, subportfolio_index):
        self.name = name
        self.subportfolio_index = subportfolio_index

    # 计算左压天数
    def calculate_zyts(self, context, stock):
        high_prices = attribute_history(stock, 101, '1d', fields=['high'], skip_paused=True)['high']
        prev_high = high_prices.iloc[-1]
        zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
        zyts = zyts_0 + 5
        return zyts

    def transform_date(self, context, date, date_type):
        if type(date) == str:
            str_date = date
            dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
            d_date = dt_date.date()
        elif type(date) == dt.datetime:
            str_date = date.strftime('%Y-%m-%d')
            dt_date = date
            d_date = dt_date.date()
        elif type(date) == dt.date:
            str_date = date.strftime('%Y-%m-%d')
            dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
            d_date = date
        dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
        return dct[date_type]

    def get_shifted_date(self, context, date, days, days_type='T'):
        # 获取上一个自然日
        d_date = self.transform_date(context, date, 'd')
        yesterday = d_date + dt.timedelta(-1)
        # 移动days个自然日
        if days_type == 'N':
            shifted_date = yesterday + dt.timedelta(days + 1)
        # 移动days个交易日
        if days_type == 'T':
            all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
            # 如果上一个自然日是交易日，根据其在交易日列表中的index计算平移后的交易日
            if str(yesterday) in all_trade_days:
                shifted_date = all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
            # 否则，从上一个自然日向前数，先找到最近一个交易日，再开始平移
            else:
                for i in range(100):
                    last_trade_date = yesterday - dt.timedelta(i)
                    if str(last_trade_date) in all_trade_days:
                        shifted_date = all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
                        break
        return str(shifted_date)

    def stockpool(self, context, pool_id=1, index=None, is_filter_kcbj=True, is_filter_st=True, is_filter_paused=True,
                  is_filter_highlimit=True,
                  is_filter_lowlimit=True, is_filter_new=True):
        log.info(self.name, '--stockpool函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if index is None:
            lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        else:
            lists = list(get_index_stocks(index))

        if pool_id == 0:
            pass
        elif pool_id == 1:
            if is_filter_kcbj:
                lists = self.filter_kcbj_stock(context, lists)
            if is_filter_st:
                lists = self.filter_st_stock(context, lists)
            if is_filter_paused:
                lists = self.filter_paused_stock(context, lists)
            if is_filter_highlimit:
                lists = self.filter_highlimit_stock(context, lists)
            if is_filter_lowlimit:
                lists = self.filter_lowlimit_stock(context, lists)
            if is_filter_new:
                lists = self.filter_new_stock(context, lists, days=375)

        return lists

    ##################################  交易函数群 ##################################

    # 开仓单只
    def open_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        now_time = context.current_dt.time()
        current_data = get_current_data()
        before_buy = dt.time(9, 30) > now_time
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
    def filter_basic_stock(self, context, stock_list):
        log.info(self.name, '--filter_basic_stock过滤股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
               and not current_data[stock].is_st
               and "ST" not in current_data[stock].name
               and "*" not in current_data[stock].name
               and "退" not in current_data[stock].name
               and not (
                    stock.startswith('4') or
                    stock.startswith('8') or
                    stock.startswith('68') or
                    stock.startswith('30')
            )
               and not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=375)

        ]
        return stock_list

    # 过滤科创北交
    def filter_kcbj_stock(self, context, stock_list):
        log.info(self.name, '--filter_kcbj_stock过滤科创北交函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 使用列表推导式过滤股票
        filtered_stock_list = [stock for stock in stock_list if not (stock.startswith('4') or
                                                                     stock.startswith('8') or
                                                                     stock.startswith('68') or
                                                                     stock.startswith('30'))]

        return filtered_stock_list

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
                not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=days)]

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

    # 计算市场宽度
    def get_market_breadth(self, context, max_industry_cnt):
        log.info(self.name, '--get_market_breadth--计算市场宽度，选择偏离程度最高的行业--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        """
        计算市场宽度，选择偏离程度最高的行业
        """
        # 指定日期以防止未来数据
        yesterday = context.previous_date

        # 获取上证中小盘指数的成分股
        stocks = get_index_stocks("000985.XSHG")

        # 获取历史收盘价数据，包括20天移动平均所需的数据
        count = 1
        historical_prices = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )

        # 将时间字段转换为日期
        historical_prices["date"] = pd.DatetimeIndex(historical_prices['time']).date

        # 将数据重塑为股票代码为索引，日期为列
        close_prices = historical_prices.pivot(index="code", columns="date", values="close")
        close_prices = close_prices.dropna(axis=0)

        # 计算20日移动平均
        ma20 = close_prices.rolling(window=20, axis=1).mean().iloc[:, -count:]

        # 获取最新一天的收盘价
        last_close_prices = close_prices.iloc[:, -count:]

        # 计算偏离程度（当前收盘价是否大于20日均线）
        bias = last_close_prices > ma20

        # 获取股票所属行业
        industries = self.getStockIndustry(stocks)
        bias["industry_name"] = industries

        # 按行业统计偏离股票的比例
        industry_bias_sum = bias.groupby("industry_name").sum()
        industry_bias_count = bias.groupby("industry_name").count()
        df_ratio = (industry_bias_sum * 100.0 / industry_bias_count).round()

        # 获取偏离比例最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(max_industry_cnt)
        top_industries = top_values.index.tolist()

        # 计算全市场宽度的平均偏离比例
        market_width = df_ratio.sum(axis=0).mean()

        log.info(
            [name for name in top_industries],
            "  全市场宽度：",
            market_width
        )

        return top_industries

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        dict = {
            stock: info["sw_l1"]["industry_name"]
            for stock, info in industry.items()
            if "sw_l1" in info
        }
        return pd.Series(dict)

    # 计算市场温度
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

    def balance_subportfolios(self, context):
        log.info(self.name, '--balance_subportfolios平衡账户资金函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位计划调整的比例："
            + str(
                g.portfolio_value_proportion
            )
            +
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        # 先把所有可用资金打入一号资金仓位
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if context.subportfolios[i].available_cash > 0 and target < value:
                transfer_cash(
                    from_pindex=i,
                    to_pindex=0,
                    cash=min(value - target, context.subportfolios[i].available_cash),
                )
        # 如果子账户仓位过低，从一号仓位往其中打入资金
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if target > value and context.subportfolios[0].available_cash > 0:
                transfer_cash(
                    from_pindex=0,
                    to_pindex=i,
                    cash=min(target - value, context.subportfolios[0].available_cash),
                )
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_small_to_other(self, context):
        current_month = context.current_dt.month
        if current_month not in (1, 4):
            return
        log.info(self.name, '--balance_subportfolios_by_small_to_other 择时资金转出--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (1, 4):
            # 把小市值仓位的资金均分到大市值和ETF
            transfer_dict = {}
            available_cash = context.subportfolios[length - 1].available_cash
            for i in range(0, length - 1):
                log.info('第', length - 1, '个仓位当前可用金额:', available_cash,
                         '，按【', context.portfolio_value_proportion[i], '】比例转到仓位', i)
                value = available_cash * (context.portfolio_value_proportion[i] / 0.5)
                transfer_dict[i] = value
                if value > 0:
                    transfer_cash(
                        from_pindex=length - 1,  ## 小市值
                        to_pindex=i,  ## 大市值 && ETF
                        cash=value
                    )
                    log.info('第', length - 1, '个仓位给第', i, '个仓位转账:', value)
            context.balance_value[current_month] = transfer_dict.copy()
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_other_to_small(self, context):
        current_month = context.current_dt.month
        if current_month not in (2, 5):
            return
        log.info(self.name, '--balance_subportfolios_by_other_to_small 择时资金转入--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (2, 5):
            # 获取上个月份
            if current_month == 2:
                last_month = 1
            else:
                last_month = 4
            # 检查是否有上个月的转账记录
            if last_month in context.balance_value:
                transfer_dict = context.balance_value[last_month]
                # 把小市值仓位的资金归还回来
                for i in transfer_dict:
                    value = transfer_dict[i]
                    if value > 0:
                        transfer_cash(
                            from_pindex=i,
                            to_pindex=length - 1,  ## 小市值
                            cash=value
                        )
                        log.info('第', i, '个仓位给第', length - 1, '个仓位转账:', value)
                # 删除上个月的转账记录
                del context.balance_value[last_month]
            else:
                log.info('没有上个月的转账记录，无需归还')
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
