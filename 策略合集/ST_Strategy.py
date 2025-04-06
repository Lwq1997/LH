# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
from jqlib.technical_analysis import *
import pandas as pd


class ST_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        st_list = []
        init_st_list = self.utilstool.get_st(context)
        # 1 4 12月 国九
        singal = self.today_is_between(context)
        if singal == True:
            print(f'筛选前市面上所有的ST股票个数：{len(init_st_list)}')
            init_st_list = self.GJT_filter_stocks(init_st_list)
            print(f'筛选后市面上所有的符合国九条ST股票个数：{len(init_st_list)}')

        init_st_list = self.st_filter_stocks(context, init_st_list)
        log.debug(f'基础信息过滤后符合条件的ST股票池：{init_st_list}')
        if len(init_st_list) == 0:
            return st_list
        init_st_list = self.st_rzq_list(context, init_st_list)
        log.debug(f'弱转强过滤后符合条件的ST股票池：{init_st_list}')
        if len(init_st_list) == 0:
            return st_list
            # 低开
        df = get_price(init_st_list, end_date=context.previous_date, frequency='daily', fields=['close'], count=1,
                       panel=False,
                       fill_paused=False, skip_paused=True).set_index('code')
        df['open_now'] = [current_data[s].day_open for s in init_st_list]
        df = df[(df['open_now'] / df['close']) < 1.01]  # 低开越多风险越大，选择3个多点即可
        df = df[(df['open_now'] / df['close']) > 0.95]
        st_list = list(df.index)
        if len(st_list) == 0:
            return st_list
        df = get_valuation(st_list, start_date=context.previous_date,
                           end_date=context.previous_date,
                           fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']
                           )
        df = df.sort_values(by='turnover_ratio', ascending=False)
        st_list = list(df.code)

        log.info('今日ST选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in st_list))

        return st_list

    def today_is_between(self, context):
        today = context.current_dt.strftime('%m-%d')
        if ('01-15' <= today) and (today <= '01-31'):
            return True
        elif ('04-15' <= today) and (today <= '04-31'):
            return True
        elif ('12-15' <= today) and (today <= '12-31'):
            return True
        else:
            return False

    ##国九条筛选##
    def GJT_filter_stocks(self, stocks):
        # 国九更新：过滤近一年净利润为负且营业收入小于1亿的
        # 国九更新：过滤近一年期末净资产为负的 (经查询没有为负数的，所以直接pass这条)
        q = query(
            valuation.code,
            valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
            income.np_parent_company_owners,  # 归属于母公司所有者的净利润
            income.net_profit,  # 净利润
            income.operating_revenue  # 营业收入
            # security_indicator.net_assets
        ).filter(
            valuation.code.in_(stocks),
            income.np_parent_company_owners > 0,
            income.net_profit > 0,
            income.operating_revenue > 1e8,
            indicator.roe > 0,
            indicator.roa > 0,
        )
        df = get_fundamentals(q)

        final_list = list(df.code)

        return final_list

    ##技术指标筛选##
    def st_filter_stocks(self, context, stocks):
        yesterday = pd.Timestamp(context.previous_date)  # 关键修改点
        df = get_price(
            stocks,
            count=11,
            frequency='1d',
            fields=['close', 'low', 'volume', 'money'],
            end_date=yesterday,
            panel=False
        ).reset_index()
        # 按股票分组处理
        grouped = df.groupby('code')
        # 计算技术指标
        ma10 = grouped['close'].transform(lambda x: x.rolling(10).mean())  # 10日均线
        prev_low = grouped['low'].shift(1)  # 前一日最低价
        prev_volume = grouped['volume'].shift(1)  # 前一日成交量
        prev_money = grouped['money'].shift(1)  # 前一日成交量
        # 构建筛选条件
        conditions = (
                (df['close'] > prev_low) &  # 多头排列
                (df['close'] > ma10) &  # 10日线上方
                (df['volume'] > prev_volume) &  # 放量
                # (df['money'] >= 10000000 ) &  # 成交量大于3000w
                (df['volume'] < 10 * prev_volume) &  # 成交量未暴增
                (df['close'] > 1)  # 股价>1
        )

        # 精准获取最新交易日数据（双重验证）
        latest_mask = (df['time'] == yesterday) & (df['time'] == df['time'].max())
        latest_data = df[latest_mask].copy()  # 创建独立副本

        # 在最新数据子集上应用条件
        final_mask = conditions[latest_mask]  # 保持索引对齐
        valid_stocks = latest_data.loc[final_mask, 'code'].unique().tolist()

        return valid_stocks

    ##筛选昨日不涨停的股票##
    def st_rzq_list(self, context, initial_list):
        # 文本日期
        date = context.previous_date

        date_2, date_1, date = get_trade_days(end_date=date, count=3)

        # 昨日不涨停
        h1_list = self.utilstool.get_ever_hl_stock3(context, initial_list, date)
        # 前日涨停过滤
        elements_to_remove = self.utilstool.get_hl_stock(context, initial_list, date_1)

        rzq_list = [stock for stock in h1_list if stock in elements_to_remove]

        return rzq_list
