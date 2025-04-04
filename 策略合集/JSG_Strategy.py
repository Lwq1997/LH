# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
from jqlib.technical_analysis import *


class JSG_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.max_industry_cnt = 1

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        top_industries = self.utilstool.get_market_breadth(context, self.max_industry_cnt)
        industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}
        if not industries.intersection(top_industries):
            # 根据市场温度设置选股条件，选出股票
            self.select_list = self.__get_rank(context)[:self.max_select_count]
        else:
            self.select_list = [self.fill_stock]
        log.info(self.name, '的选股列表:', self.select_list)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        initial_list = super().stockpool(context, 1, "399101.XSHE")

        q = query(
            valuation.code,
        ).filter(
            valuation.code.in_(initial_list),
            income.np_parent_company_owners > 0,  # 归属于母公司所有者的净利润大于0
            income.net_profit > 0,  # 净利润大于0
            income.operating_revenue > 1e8  # 营业收入大于1亿
        ).order_by(
            valuation.market_cap.asc()
        ).limit(self.max_select_count)

        final_stocks = list(get_fundamentals(q).code)
        return final_stocks
