# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入

from kuanke.user_space_api import *
from kuanke.wizard import *
from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
from 策略合集.XSZYZ_Strategy import XSZYZ_Strategy


def initialize(context):
    log.warn('--initialize函数(只运行一次)--',
             str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    # 关闭未来函数
    set_option('avoid_future_data', True)

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0001, close_commission=0.0001, min_commission=0),
                   type='stock')

    # 为股票设定滑点为百分比滑点
    set_slippage(PriceRelatedSlippage(0.01), type='stock')

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                    0, 0, 0]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[4], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[5], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[6], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[7], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[8], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[9], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[10], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[11], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[12], 'stock')
    ])

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0
    factor_list = [
        (  # ARBR-SGAI-NPTTOR-RPPS.txt
            [
                'ARBR',  # 情绪类因子 ARBR
                'SGAI',  # 质量类因子 销售管理费用指数
                'net_profit_to_total_operate_revenue_ttm',  # 质量类因子 净利润与营业总收入之比
                'retained_profit_per_share'  # 每股指标因子 每股未分配利润
            ],
            [
                -2.3425,
                -694.7936,
                -170.0463,
                -1362.5762
            ]
        ),
        (  # FL-VOL240-AEttm.txt
            [
                'financial_liability',
                'VOL240',
                'administration_expense_ttm'
            ],
            [
                -5.305338739321596e-13,
                0.0028018907262207246,
                3.445005190225511e-13
            ]
        ),
        (  # ITR-StPR-STM-NLoMC.txt
            [
                'inventory_turnover_rate',
                'sales_to_price_ratio',
                'share_turnover_monthly',
                'natural_log_of_market_cap'
            ],
            [
                2.758707919895875e-08, 0.02830291416983057, -0.033608724791129085, 0.0013219161779863542
            ]
        ),
        (  # Liquidity-VCPttm-ROAttm.txt
            [
                'liquidity',  # 风格因子 流动性因子
                'value_change_profit_ttm',  # 基础科目及衍生类因子 价值变动净收益TTM
                'roa_ttm'  # 质量类因子 资产回报率TTM
            ],
            [
                -0.04963427582597701,
                6.451436607157746e-13,
                -0.04698060391789672
            ]
        ),
        (  # NCAR-AER-ATR6-VOL20.txt
            [
                'non_current_asset_ratio',  # 质量类因子 非流动资产比率
                'admin_expense_rate',  # 质量类因子 管理费用与营业总收入之比
                'ATR6',  # 情绪类因子 6日均幅指标
                'VOL20'  # 情绪类因子 20日平均换手率
            ],
            [238.1242, -347.1289, 4.2208, -19.8349
             ]
        ),
        (  # ORGR-SRFps-VSTD20-NOCFtOI.txt
            [
                'operating_revenue_growth_rate',  # 成长类因子 营业收入增长率
                'surplus_reserve_fund_per_share',  # 每股指标因子 每股盈余公积金
                'VSTD20',  # 情绪类因子 20日成交量标准差
                'net_operate_cash_flow_to_operate_income',  # 质量类因子 经营活动产生的现金流量净额与经营活动净收益之比
            ],
            [
                -2.2611191074512323e-05,
                -0.007031472339507336,
                -2.2140446594154373e-10,
                6.483698165689134e-05
            ]
        ),
        (  # ORGR-TPGR-NPGR-EGR-EPS.txt
            [
                'operating_revenue_growth_rate',  # 成长类因子 营业收入增长率
                'total_profit_growth_rate',  # 成长类因子 利润总额增长率
                'net_profit_growth_rate',  # 成长类因子 净利润增长率
                'earnings_growth',  # 风格因子 5年盈利增长率
                'eps_ttm'  # 每股指标因子 每股收益TTM
            ],
            [
                -0.0019079645149417137, -6.027115922691245e-05, -1.8580428418195642e-05, -0.005293892163117587,
                -0.010077397467005972
            ]
        ),
        (  # PNF-TPtCR-ITR.txt
            [
                'price_no_fq',  # 技术指标因子 不复权价格因子
                'total_profit_to_cost_ratio',  # 质量类因子 成本费用利润率
                'inventory_turnover_rate'  # 质量类因子 存货周转率
            ],
            [
                -6.123355346008858e-05,
                -0.002579342458393642,
                -2.194257357346814e-06
            ]
        ),
        (  # SQR-CoS-CtE.txt
            [
                'super_quick_ratio',  # 质量类因子 超速动比率
                'cube_of_size',  # 风险因子 市值立方
                'cfo_to_ev'  # 质量类因子 经营活动产生的现金流量净额与企业价值之比TTM
            ],
            [
                -26.6636, -2.6880, 1242.3598
            ]
        ),
        (  # VSTD20-ARTR-LTDtAR-OC.txt
            [
                'VSTD20',  # 情绪类因子 20日成交量标准差
                'account_receivable_turnover_rate',  # 质量类因子 应收账款周转率
                'long_term_debt_to_asset_ratio',  # 质量类因子 长期负债与资产总计之比
                'OperatingCycle'  # 质量类因子 营业周期
            ],
            [
                -1.3783e-09, -4.8282e-16, -4.6013e-02, 2.4878e-09
            ]
        )
        , (  # P1Y-TPtCR-VOL120
            [
                'Price1Y',  # 动量类因子 当前股价除以过去一年股价均值再减1
                'total_profit_to_cost_ratio',  # 质量类因子 成本费用利润率
                'VOL120'  # 情绪类因子 120日平均换手率
            ],
            [
                -0.0647128120839873,
                -0.006385116279168804,
                -0.0029867925845833217
            ]
        ),
        (  # DtA-OCtORR-DAVOL20-PNF-SG
            [
                'debt_to_assets',  # 风格因子 资产负债率
                'operating_cost_to_operating_revenue_ratio',  # 质量类因子 销售成本率
                'DAVOL20',  # 情绪类因子 20日平均换手率与120日平均换手率之比
                'price_no_fq',  # 技术指标因子 不复权价格因子
                'sales_growth'  # 风格因子 5年营业收入增长率
            ],
            [
                0.04477354820057883,
                0.021636407482421707,
                -0.01864268317469762,
                -0.0004678118383947827,
                0.02884867440332058
            ]
        ),
        (  # TVSTD6-CFpsttm-SR120-NONPttm
            [
                'TVSTD6',  # 情绪类因子 6日成交金额的标准差
                'cashflow_per_share_ttm',  # 每股指标因子 每股现金流量净额
                'sharpe_ratio_120',  # 风险类因子 120日夏普率
                'non_operating_net_profit_ttm'  # 基础科目及衍生类因子 营业外收支净额TTM
            ],
            [
                -5.394060941494863e-12,
                4.6306072704138405e-05,
                -0.0030567075906980912,
                1.4227113275455325e-12
            ]
        )
    ]

    for i in range(12):
        params = {
            'max_hold_count': 500,  # 最大持股数
            'max_select_count': 100,  # 最大输出选股数
            'per_factor_max_select_count': 1,  # 最大输出选股数
            'factor_list': [list(factor_list[i])]
        }
        xszyz_strategy = XSZYZ_Strategy(context, subportfolio_index=i, name='小市值聚宽因子策略_' + str(i),
                                        params=params)
        g.strategys["小市值聚宽因子策略_" + str(i)] = xszyz_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    if g.portfolio_value_proportion[0] > 0:
        run_daily(xszyz_day_prepare_0, time='09:00')
        # 选股
        run_daily(xszyz_select_0, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_0, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_0, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_0, 'after_close')

    if g.portfolio_value_proportion[1] > 0:
        run_daily(xszyz_day_prepare_1, time='09:00')
        # 选股
        run_daily(xszyz_select_1, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_1, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_1, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_1, 'after_close')

    if g.portfolio_value_proportion[2] > 0:
        run_daily(xszyz_day_prepare_2, time='09:00')
        # 选股
        run_daily(xszyz_select_2, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_2, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_2, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_2, 'after_close')

    if g.portfolio_value_proportion[3] > 0:
        run_daily(xszyz_day_prepare_3, time='09:00')
        # 选股
        run_daily(xszyz_select_3, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_3, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_3, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_3, 'after_close')

    if g.portfolio_value_proportion[4] > 0:
        run_daily(xszyz_day_prepare_4, time='09:00')
        # 选股
        run_daily(xszyz_select_4, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_4, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_4, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_4, 'after_close')

    if g.portfolio_value_proportion[5] > 0:
        run_daily(xszyz_day_prepare_5, time='09:00')
        # 选股
        run_daily(xszyz_select_5, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_5, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_5, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_5, 'after_close')

    if g.portfolio_value_proportion[6] > 0:
        run_daily(xszyz_day_prepare_6, time='09:00')
        # 选股
        run_daily(xszyz_select_6, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_6, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_6, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_6, 'after_close')

    if g.portfolio_value_proportion[7] > 0:
        run_daily(xszyz_day_prepare_7, time='09:00')
        # 选股
        run_daily(xszyz_select_7, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_7, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_7, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_7, 'after_close')

    if g.portfolio_value_proportion[8] > 0:
        run_daily(xszyz_day_prepare_8, time='09:00')
        # 选股
        run_daily(xszyz_select_8, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_8, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_8, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_8, 'after_close')

    if g.portfolio_value_proportion[9] > 0:
        run_daily(xszyz_day_prepare_9, time='09:00')
        # 选股
        run_daily(xszyz_select_9, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_9, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_9, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_9, 'after_close')

    if g.portfolio_value_proportion[10] > 0:
        run_daily(xszyz_day_prepare_10, time='09:00')
        # 选股
        run_daily(xszyz_select_10, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_10, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_10, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_10, 'after_close')

    if g.portfolio_value_proportion[11] > 0:
        run_daily(xszyz_day_prepare_11, time='09:00')
        # 选股
        run_daily(xszyz_select_11, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_11, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_11, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_11, 'after_close')

    if g.portfolio_value_proportion[12] > 0:
        run_daily(xszyz_day_prepare_12, time='09:00')
        # 选股
        run_daily(xszyz_select_12, time='09:31')
        # 卖出
        run_daily(xszyz_adjust_12, time='09:31')
        # 非涨停卖出
        run_daily(xszyz_sell_when_highlimit_open_12, time='14:00')
        # 收盘
        run_daily(xszyz_after_market_close_12, 'after_close')


def xszyz_day_prepare_0(context):
    g.strategys["小市值聚宽因子策略_0"].day_prepare(context)


def xszyz_select_0(context):
    g.strategys["小市值聚宽因子策略_0"].select(context)


def xszyz_adjust_0(context):
    g.strategys["小市值聚宽因子策略_0"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_0(context):
    g.strategys["小市值聚宽因子策略_0"].sell_when_highlimit_open(context)


def xszyz_after_market_close_0(context):
    g.strategys["小市值聚宽因子策略_0"].after_market_close(context)


def xszyz_day_prepare_1(context):
    g.strategys["小市值聚宽因子策略_1"].day_prepare(context)


def xszyz_select_1(context):
    g.strategys["小市值聚宽因子策略_1"].select(context)


def xszyz_adjust_1(context):
    g.strategys["小市值聚宽因子策略_1"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_1(context):
    g.strategys["小市值聚宽因子策略_1"].sell_when_highlimit_open(context)


def xszyz_after_market_close_1(context):
    g.strategys["小市值聚宽因子策略_1"].after_market_close(context)


def xszyz_day_prepare_2(context):
    g.strategys["小市值聚宽因子策略_2"].day_prepare(context)


def xszyz_select_2(context):
    g.strategys["小市值聚宽因子策略_2"].select(context)


def xszyz_adjust_2(context):
    g.strategys["小市值聚宽因子策略_2"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_2(context):
    g.strategys["小市值聚宽因子策略_2"].sell_when_highlimit_open(context)


def xszyz_after_market_close_2(context):
    g.strategys["小市值聚宽因子策略_2"].after_market_close(context)


def xszyz_day_prepare_3(context):
    g.strategys["小市值聚宽因子策略_3"].day_prepare(context)


def xszyz_select_3(context):
    g.strategys["小市值聚宽因子策略_3"].select(context)


def xszyz_adjust_3(context):
    g.strategys["小市值聚宽因子策略_3"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_3(context):
    g.strategys["小市值聚宽因子策略_3"].sell_when_highlimit_open(context)


def xszyz_after_market_close_3(context):
    g.strategys["小市值聚宽因子策略_3"].after_market_close(context)


def xszyz_day_prepare_4(context):
    g.strategys["小市值聚宽因子策略_4"].day_prepare(context)


def xszyz_select_4(context):
    g.strategys["小市值聚宽因子策略_4"].select(context)


def xszyz_adjust_4(context):
    g.strategys["小市值聚宽因子策略_4"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_4(context):
    g.strategys["小市值聚宽因子策略_4"].sell_when_highlimit_open(context)


def xszyz_after_market_close_4(context):
    g.strategys["小市值聚宽因子策略_4"].after_market_close(context)


def xszyz_day_prepare_5(context):
    g.strategys["小市值聚宽因子策略_5"].day_prepare(context)


def xszyz_select_5(context):
    g.strategys["小市值聚宽因子策略_5"].select(context)


def xszyz_adjust_5(context):
    g.strategys["小市值聚宽因子策略_5"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_5(context):
    g.strategys["小市值聚宽因子策略_5"].sell_when_highlimit_open(context)


def xszyz_after_market_close_5(context):
    g.strategys["小市值聚宽因子策略_5"].after_market_close(context)


def xszyz_day_prepare_6(context):
    g.strategys["小市值聚宽因子策略_6"].day_prepare(context)


def xszyz_select_6(context):
    g.strategys["小市值聚宽因子策略_6"].select(context)


def xszyz_adjust_6(context):
    g.strategys["小市值聚宽因子策略_6"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_6(context):
    g.strategys["小市值聚宽因子策略_6"].sell_when_highlimit_open(context)


def xszyz_after_market_close_6(context):
    g.strategys["小市值聚宽因子策略_6"].after_market_close(context)


def xszyz_day_prepare_7(context):
    g.strategys["小市值聚宽因子策略_7"].day_prepare(context)


def xszyz_select_7(context):
    g.strategys["小市值聚宽因子策略_7"].select(context)


def xszyz_adjust_7(context):
    g.strategys["小市值聚宽因子策略_7"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_7(context):
    g.strategys["小市值聚宽因子策略_7"].sell_when_highlimit_open(context)


def xszyz_after_market_close_7(context):
    g.strategys["小市值聚宽因子策略_7"].after_market_close(context)


def xszyz_day_prepare_8(context):
    g.strategys["小市值聚宽因子策略_8"].day_prepare(context)


def xszyz_select_8(context):
    g.strategys["小市值聚宽因子策略_8"].select(context)


def xszyz_adjust_8(context):
    g.strategys["小市值聚宽因子策略_8"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_8(context):
    g.strategys["小市值聚宽因子策略_8"].sell_when_highlimit_open(context)


def xszyz_after_market_close_8(context):
    g.strategys["小市值聚宽因子策略_8"].after_market_close(context)


def xszyz_day_prepare_9(context):
    g.strategys["小市值聚宽因子策略_9"].day_prepare(context)


def xszyz_select_9(context):
    g.strategys["小市值聚宽因子策略_9"].select(context)


def xszyz_adjust_9(context):
    g.strategys["小市值聚宽因子策略_9"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_9(context):
    g.strategys["小市值聚宽因子策略_9"].sell_when_highlimit_open(context)


def xszyz_after_market_close_9(context):
    g.strategys["小市值聚宽因子策略_9"].after_market_close(context)


def xszyz_day_prepare_10(context):
    g.strategys["小市值聚宽因子策略_10"].day_prepare(context)


def xszyz_select_10(context):
    g.strategys["小市值聚宽因子策略_10"].select(context)


def xszyz_adjust_10(context):
    g.strategys["小市值聚宽因子策略_10"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_10(context):
    g.strategys["小市值聚宽因子策略_10"].sell_when_highlimit_open(context)


def xszyz_after_market_close_10(context):
    g.strategys["小市值聚宽因子策略_10"].after_market_close(context)


def xszyz_day_prepare_11(context):
    g.strategys["小市值聚宽因子策略_11"].day_prepare(context)


def xszyz_select_11(context):
    g.strategys["小市值聚宽因子策略_11"].select(context)


def xszyz_adjust_11(context):
    g.strategys["小市值聚宽因子策略_11"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_11(context):
    g.strategys["小市值聚宽因子策略_11"].sell_when_highlimit_open(context)


def xszyz_after_market_close_11(context):
    g.strategys["小市值聚宽因子策略_11"].after_market_close(context)


def xszyz_day_prepare_12(context):
    g.strategys["小市值聚宽因子策略_12"].day_prepare(context)


def xszyz_select_12(context):
    g.strategys["小市值聚宽因子策略_12"].select(context)


def xszyz_adjust_12(context):
    g.strategys["小市值聚宽因子策略_12"].adjustwithnoRM(context)


def xszyz_sell_when_highlimit_open_12(context):
    g.strategys["小市值聚宽因子策略_12"].sell_when_highlimit_open(context)


def xszyz_after_market_close_12(context):
    g.strategys["小市值聚宽因子策略_12"].after_market_close(context)
