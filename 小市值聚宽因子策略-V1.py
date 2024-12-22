# 克隆自聚宽文章：https://www.joinquant.com/post/51093
# 标题：纯娱乐，猜猜小盘会 怎么蹦了
# 作者：吾生不负韶华

# 克隆自聚宽文章：https://www.joinquant.com/post/40981
# 标题：差不多得了
# 作者：wywy1995

# 克隆自聚宽文章：https://www.joinquant.com/post/40407
# 标题：wywy1995大侠的小市值AI因子选股 5组参数50股测试
# 作者：Bruce_Lee

# https://www.joinquant.com/view/community/detail/30684f8d65a74eef0d704239f0eec8be?type=1&page=2
# 导入函数库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd


# 初始化函数
def initialize(context):
    # 设定基准
    set_benchmark('000905.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化全局变量
    g.no_trading_today_signal = False
    g.stock_num = 1
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    g.factor_list = [
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
        (  #ITR-StPR-STM-NLoMC.txt
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
        (  #Liquidity-VCPttm-ROAttm.txt
            [
                'liquidity', #风格因子 流动性因子
                'value_change_profit_ttm', #基础科目及衍生类因子 价值变动净收益TTM
                'roa_ttm' #质量类因子 资产回报率TTM
            ],
            [
                -0.04963427582597701,
                6.451436607157746e-13,
                -0.04698060391789672
            ]
        ),
        (  #NCAR-AER-ATR6-VOL20.txt
            [
                'non_current_asset_ratio', #质量类因子 非流动资产比率
                'admin_expense_rate', #质量类因子 管理费用与营业总收入之比
                'ATR6', #情绪类因子 6日均幅指标
                'VOL20' #情绪类因子 20日平均换手率
            ],
            [238.1242, -347.1289, 4.2208,  -19.8349
            ]
        ),
        (  #ORGR-SRFps-VSTD20-NOCFtOI.txt
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
        (  #ORGR-TPGR-NPGR-EGR-EPS.txt
            [
                'operating_revenue_growth_rate',  # 成长类因子 营业收入增长率
                'total_profit_growth_rate',  # 成长类因子 利润总额增长率
                'net_profit_growth_rate',  # 成长类因子 净利润增长率
                'earnings_growth',  # 风格因子 5年盈利增长率
                'eps_ttm'  # 每股指标因子 每股收益TTM
            ],
            [
                -0.0019079645149417137, -6.027115922691245e-05, -1.8580428418195642e-05, -0.005293892163117587, -0.010077397467005972
            ]
        ),
        (  #PNF-TPtCR-ITR.txt
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
        (  #SQR-CoS-CtE.txt
            [
                'super_quick_ratio',  # 质量类因子 超速动比率
                'cube_of_size',  # 风险因子 市值立方
                'cfo_to_ev'  # 质量类因子 经营活动产生的现金流量净额与企业价值之比TTM
            ],
            [
                -26.6636, -2.6880, 1242.3598
            ]
        ),
        (  #VSTD20-ARTR-LTDtAR-OC.txt
            [
                'VSTD20',  # 情绪类因子 20日成交量标准差
                'account_receivable_turnover_rate',  # 质量类因子 应收账款周转率
                'long_term_debt_to_asset_ratio',  # 质量类因子 长期负债与资产总计之比
                'OperatingCycle'  # 质量类因子 营业周期
            ],
            [
                -1.3783e-09, -4.8282e-16, -4.6013e-02, 2.4878e-09
            ]
        ),
        (  # P1Y-TPtCR-VOL120
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
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_weekly(weekly_adjustment, 1, '9:31')
    run_daily(check_limit_up, '14:00')  # 检查持仓中的涨停股是否需要卖出
    run_daily(close_account, '14:30')
    run_daily(print_position_info, '15:10')


# 1-1 准备股票池
def prepare_stock_list(context):
    # 获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    # 获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit'],
                       count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
    # 判断今天是否为账户资金再平衡的日期
    g.no_trading_today_signal = today_is_between(context, '04-05', '04-30')


# 1-2 选股模块
def get_stock_list(context):
    # 指定日期防止未来数据
    yesterday = context.previous_date
    today = context.current_dt
    # 获取初始列表
    initial_list = get_all_securities('stock', today).index.tolist()
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    final_list = []
    # MS
    for factor_list, coef_list in g.factor_list:
        factor_values = get_factor_values(initial_list, factor_list, end_date=yesterday, count=1)
        df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
        for i in range(len(factor_list)):
            df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:, 0])
        df = df.dropna()
        df['total_score'] = 0
        for i in range(len(factor_list)):
            df['total_score'] += coef_list[i] * df[factor_list[i]]
        # 按照因子*因子比例计算总分
        df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
        complex_factor_list = list(df.index)[:int(0.1 * len(list(df.index)))]
        q = query(valuation.code, valuation.circulating_market_cap, indicator.eps).filter(
            valuation.code.in_(complex_factor_list)).order_by(valuation.circulating_market_cap.asc())
        df = get_fundamentals(q)
        df = df[df['eps'] > 0]
        lst = list(df.code)
        lst = filter_paused_stock(lst)
        lst = filter_limitup_stock(context, lst)
        lst = filter_limitdown_stock(context, lst)
        lst = lst[:min(g.stock_num, len(lst))]
        log.error('factor_list:', factor_list, '选股列表:', lst)
        for stock in lst:
            if stock not in final_list:
                final_list.append(stock)
    return final_list


# 1-3 整体调整持仓
def weekly_adjustment(context):
    if g.no_trading_today_signal == False:
        # 获取应买入列表
        target_list = get_stock_list(context)
        log.error('最终选股列表:', target_list)
        # 调仓卖出
        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                log.info("卖出[%s]" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("已持有[%s]" % (stock))
        # 调仓买入
        position_count = len(context.portfolio.positions)
        target_num = len(target_list)
        if target_num > position_count:
            value = context.portfolio.cash / (target_num - position_count)
            for stock in target_list:
                if context.portfolio.positions[stock].total_amount == 0:
                    if open_position(stock, value):
                        if len(context.portfolio.positions) == target_num:
                            break


# 1-4 调整昨日涨停股票
def check_limit_up(context):
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("[%s]涨停，继续持有" % (stock))


# 2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


# 2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
            stock_list.remove(stock)
    return stock_list


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if
            not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]


# 3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)


# 3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False


# 3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


# 4-1 判断今天是否为账户资金再平衡的日期
def today_is_between(context, start_date, end_date):
    today = context.current_dt.strftime('%m-%d')
    if (start_date <= today) and (today <= end_date):
        return True
    else:
        return False


# 4-2 清仓后次日资金可转
def close_account(context):
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0:
            for stock in g.hold_list:
                position = context.portfolio.positions[stock]
                close_position(position)
                log.info("卖出[%s]" % (stock))


# 4-3 打印每日持仓信息
def print_position_info(context):
    # 打印当天成交记录
    trades = get_trades()
    for _trade in trades.values():
        print('成交记录：' + str(_trade))
    # 打印账户信息
    for position in list(context.portfolio.positions.values()):
        securities = position.security
        cost = position.avg_cost
        price = position.price
        ret = 100 * (price / cost - 1)
        value = position.value
        amount = position.total_amount
        print('代码:{}'.format(securities))
        print('成本价:{}'.format(format(cost, '.2f')))
        print('现价:{}'.format(price))
        print('收益率:{}%'.format(format(ret, '.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value, '.2f')))
        print('———————————————————————————————————')
    print('———————————————————————————————————————分割线————————————————————————————————————————')
