#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
from jqlib.technical_analysis import *


# DSZMX策略
class DSZ_MX_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)


    def select(self, context):
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        lists = self.stockpool(context)

        # 基本股选股
        q = query(
            valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        ).filter(
            valuation.code.in_(lists),
            valuation.pe_ratio_lyr.between(0, 30),  # 市盈率
            valuation.ps_ratio.between(0, 8),  # 市销率TTM
            valuation.pcf_ratio < 10,  # 市现率TTM
            indicator.eps > 0.3,  # 每股收益
            indicator.roe > 0.1,  # 净资产收益率
            indicator.net_profit_margin > 0.1,  # 销售净利率
            indicator.gross_profit_margin > 0.3,  # 销售毛利率
            indicator.inc_revenue_year_on_year > 0.25  # 营业收入同比增长率
        ).order_by(
            valuation.market_cap.desc()
        ).limit(
            self.max_select_count * 3
        )
        lists = list(get_fundamentals(q).code)
        return lists
