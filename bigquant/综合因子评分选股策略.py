from bigmodule import M
from bigtrader.finance.commission import PerOrder


# <aistudiograph>

# @param(id="m3", name="initialize")
# 回测引擎：初始化函数，只执行一次
def m3_initialize_bigquant_run(context):
    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    context.stock_count = 10

    # 每只股票的权重平均分配
    context.stock_weights = 1 / context.stock_count

    # 每日换仓数
    context.change_num = 1

    # 剔除科创板
    context.my_data = context.data[~context.data.instrument.str.contains('688')]
    context.my_data = context.my_data.sort_values(["date", "position"])


# @param(id="m3", name="before_trading_start")
# 交易引擎：每个单位时间开盘前调用一次。
def m3_before_trading_start_bigquant_run(context, data):
    # 盘前处理，订阅行情等
    pass


# @param(id="m3", name="handle_tick")
# 交易引擎：tick数据处理函数，每个tick执行一次
def m3_handle_tick_bigquant_run(context, tick):
    pass


# @param(id="m3", name="handle_data")
from datetime import timedelta, datetime


# 回测引擎：每日数据处理函数，每天执行一次
def m3_handle_data_bigquant_run(context, data):
    today = data.current_dt.strftime('%Y-%m-%d')
    ranker_prediction = context.my_data[context.my_data.date == today]
    # 获取当前持仓股票
    stock_now = {e: p for e, p in context.portfolio.positions.items() if p.amount > 0}
    stock_now_num = len(stock_now);
    # 当日应该买入股票
    try:
        # 多取n只应对买不了的异常情况
        buy_list = ranker_prediction.instrument.unique()[:context.stock_count + 3]
    except:
        buy_list = []

    sell_num = 0
    sell_already_lst = []
    if len(stock_now) > 0:
        # 首先卖出不在预测集的 例如st了
        need_sell = [x for x in stock_now if x not in list(ranker_prediction.instrument.unique())]
        for instrument in need_sell:
            rv = context.order_target(instrument, 0)
            if rv != 0:
                print(f"{instrument} 不在预测集卖出失败 {context.get_error_msg(rv)}")
                continue
            sell_num += 1

        # 持有的票按照预测得分排序，卖出得分低的
        instruments = list(
            reversed(list(ranker_prediction.instrument[ranker_prediction.instrument.apply(lambda x: x in stock_now)])))
        for instrument in instruments:
            if sell_num < context.change_num:
                rv = context.order_target(instrument, 0)
                if rv != 0:
                    print(f"{instrument} 买入失败 {context.get_error_msg(rv)}")
                    continue
                sell_already_lst.append(instrument)
                #                 print("普通卖出：",instrument)
                sell_num += 1

                # 有卖出就买入
    if len(buy_list) > 0 and stock_now_num < context.stock_count:
        buy_instruments = [i for i in buy_list if i not in stock_now]
        # 等权重买入，现金不够系统会自动调整
        cash_now = context.portfolio.cash / (context.stock_count - stock_now_num)
        cash_for_buy = min(context.portfolio.portfolio_value * context.stock_weights, cash_now)
        for instrument in buy_instruments:
            if stock_now_num < context.stock_count:
                rv = context.order_value(instrument, cash_for_buy)
                if rv != 0:
                    print(f"{instrument} 买入失败 {context.get_error_msg(rv)}")
                    continue
                stock_now_num = stock_now_num + 1


# @param(id="m3", name="handle_trade")
# 交易引擎：成交回报处理函数，每个成交发生时执行一次
def m3_handle_trade_bigquant_run(context, trade):
    pass


# @param(id="m3", name="handle_order")
# 交易引擎：委托回报处理函数，每个委托变化时执行一次
def m3_handle_order_bigquant_run(context, order):
    pass


# @param(id="m3", name="after_trading")
# 交易引擎：盘后处理函数，每日盘后执行一次
def m3_after_trading_bigquant_run(context, data):
    pass


# @module(position="-237,-357", comment="""通过SQL调用数据、因子和表达式等构建策略逻辑""", comment_collapsed=True)
m1 = M.input_features_dai.v6(
    sql="""-- 使用DAI SQL获取数据，构建因子等，如下是一个例子作为参考
-- DAI SQL 语法: https://bigquant.com/wiki/doc/dai-PLSbc1SbZX#h-sql%E5%85%A5%E9%97%A8%E6%95%99%E7%A8%8B

SELECT
    position,
    -- 日期，这是每个股票每天的数据
    date,
    -- 股票代码，代表每一支股票
    instrument
FROM user_factor_0c26192ef58c11eead9c3afa1a581da0
""",
    m_cached=False,
    m_name="""m1"""
)

# @module(position="-234,-280", comment="""抽取数据，设置数据开始时间和结束时间，并绑定模拟交易""", comment_collapsed=True)
m2 = M.extract_data_dai.v7(
    sql=m1.data,
    start_date="""2021-08-23""",
    start_date_bound_to_trading_date=True,
    end_date="""2022-03-29""",
    end_date_bound_to_trading_date=True,
    before_start_days=10,
    debug=False,
    m_cached=False,
    m_name="""m2"""
)

# @module(position="-311.5376510620117,-201.5376434326172", comment="""""", comment_collapsed=True)
m3 = M.bigtrader.v14(
    data=m2.data,
    start_date="""""",
    end_date="""""",
    initialize=m3_initialize_bigquant_run,
    before_trading_start=m3_before_trading_start_bigquant_run,
    handle_tick=m3_handle_tick_bigquant_run,
    handle_data=m3_handle_data_bigquant_run,
    handle_trade=m3_handle_trade_bigquant_run,
    handle_order=m3_handle_order_bigquant_run,
    after_trading=m3_after_trading_bigquant_run,
    capital_base=1000000,
    frequency="""daily""",
    product_type="""股票""",
    before_start_days=0,
    volume_limit=1,
    order_price_field_buy="""open""",
    order_price_field_sell="""close""",
    benchmark="""000300.SH""",
    plot_charts=True,
    disable_cache=False,
    debug=False,
    backtest_only=False,
    m_cached=False,
    m_name="""m3"""
)
# </aistudiograph>