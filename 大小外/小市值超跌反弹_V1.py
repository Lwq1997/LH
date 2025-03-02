# 克隆自聚宽文章：https://www.joinquant.com/post/53477
# 标题：小市值叠加行业轮动，超跌博反弹-7年115倍
# 作者：God is a pig

# 克隆自聚宽文章：https://www.joinquant.com/post/51521
# 标题：小市值调整持股数量和止损——加入指数MACD顶背离
# 作者：天才晓

# 克隆自聚宽文章：https://www.joinquant.com/post/47527
# 标题：冷饭热吃之三，14年至今年华99.99%
# 作者：韶华不负

# 克隆自聚宽文章：https://www.joinquant.com/post/47454
# 标题：策略年化收益 94.81%，最大回撤  33.50%
# 作者：李精荠

"""
策略逻辑，
1，周二1030(最OK)盘前选股(市值升序),7>10=5只
2，有4月空仓(加1月空仓)
3，2点有破板卖出
4，10点有止损卖出(中小指日跌幅6点，单票12点止损,最佳)

"""

# 导入函数库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
from datetime import time


# import datetime
# 初始化函数
def initialize(context):
  # 开启防未来函数
  # set_option('avoid_future_data', True)
  # 微盘基准
  g.wp_benchmark = '399101.XSHE'
  # 设定基准
  set_benchmark(g.wp_benchmark)
  # 用真实价格交易
  set_option('use_real_price', True)
  # 将滑点设置为0
  set_slippage(FixedSlippage(0.0003))
  # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
  set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=2.5 / 10000, close_commission=2.5 / 10000,
                           close_today_commission=0, min_commission=5), type='stock')
  # 过滤order中低于error级别的日志
  log.set_level('order', 'error')
  log.set_level('system', 'error')
  log.set_level('strategy', 'debug')


def after_code_changed(context):
  unschedule_all()  # 取消所有定时运行
  # 初始化全局变量 bool
  g.no_trading_today_signal = False  # 是否为可交易日
  g.filter_new_gj = True  # 是否过滤新国九

  # 全局变量list
  g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
  g.target_list = []
  g.stk_pool = []
  g.empty_stks = []

  g.ind_ld = True  # 是否行业轮动
  # 全局变量float/strs
  g.stock_num = 3
  # g.m_days = 7 #取值参考天数,未生效
  g.reason_to_sell = ''
  g.stoploss_strategy = 3  # 1为止损线止损，2为市场趋势止损, 3为联合1、2策略
  g.stoploss_limit = 0.88  # 止损线
  g.stoploss_market = 0.94  # 市场趋势止损参数
  g.dbl = []  # 市场顶背离list
  g.dbl_days = 10  # 市场顶背离天数

  g.HV_control = True  # 新增，Ture是日频判断是否放量，False则不然
  g.HV_duration = 20  # HV_control用，周期可以是240-120-60，默认比例是0.9
  g.HV_ratio = 2  # HV_control用
  # 设置交易运行时间
  # run_weekly(prepare_stk_pool, 1, '9:00')
  run_daily(prepare_stock_list, '9:05')
  run_daily(dapan, '9:30')
  run_weekly(adjust_position, 1, '9:40')
  run_daily(check_morning, time='10:00')  # 止损函数
  run_daily(check_afternoon, time='14:30')  # 检查持仓中的涨停股是否需要卖出
  run_daily(close_account, '14:50')


def prepare_stk_pool(context):
  initial_list = []
  if g.ind_ld:
    ind_mom_df = get_industries_mom(context.previous_date)
    ind_mom_codes = ind_mom_df[ind_mom_df['change_pct'] < 0]['code'][-3:].tolist()
    for ind_code in ind_mom_codes:
      initial_list.extend(get_industry_stocks(ind_code, context.previous_date))
  else:
    initial_list = get_index_stocks(g.wp_benchmark)

  g.stk_pool = filter_basic_pool(context.previous_date, initial_list)

  if g.filter_new_gj:
    # 国九条：财务造假退市指标：==> 【C罗：按审计无保留意见过滤】
    g.stk_pool = filter_stocks_by_auditor_opinion_jqz(context, g.stk_pool)
    # 国九条：利润总额、净利润、扣非净利润三者孰低为负值，且营业收入低于 3 亿元 退市
    g.stk_pool = filter_stocks_by_revenue_and_profit(context, g.stk_pool)
    # 国九条：最近三个会计年度累计现金分红总额低于年均净利润的30%，且累计分红金额低于5000万元的，将被实施ST
    # ==> 抓紧派利息，可能不用剔除；但实测剔除的收益更高，可能是因为派息预期推高股价
    g.stk_pool = get_dividend_ratio_filter_list(context, g.stk_pool, False, 0, 1)

  # print('initial_list中含有{}个元素'.format(len(initial_list)))
  g.stk_pool = get_fundamentals(query(valuation.code,
                                      valuation.market_cap)
                                .filter(valuation.code.in_(g.stk_pool),
                                        valuation.market_cap > 5)
                                .order_by(valuation.market_cap.asc())
                                .limit(100))['code'].tolist()
  return g.stk_pool


# 0-0 大盘顶背离函数======================================================================
def macd_gold_dead(stock, context):
  suit = {'dif': 0, 'dea': 0, 'macd': 0, 'gold': False, 'dead': False}
  # macd_df = attribute_history(stock, (fast + slow + sign) * 5, fields=['close']).dropna()
  # print(MyTT.MACD(macd_df.close,SHORT=12,LONG=26,M=9))
  # print(macd_df)
  try:
    macd_df = calc_macd(context, stock)
    # print(macd_df)

    # 底背离----------------------------------------------------------------
    mask = macd_df['macd'] > 0
    # print(mask)
    mask = mask[mask][mask.shift(1) == False]
    key2 = mask.keys()[-2]
    key1 = mask.keys()[-1]
    suit['gold'] = macd_df.close[key2] > macd_df.close[key1] and \
                   macd_df.dif[key2] < macd_df.dif[key1] < 0 and \
                   macd_df.macd[-2] < 0 < macd_df.macd[-1]
    # 顶背离----------------------------------------------------------------
    mask = macd_df['macd'] < 0
    mask = mask[mask][mask.shift(1) == False]
    key2 = mask.keys()[-2]
    key1 = mask.keys()[-1]
    # print(macd_df.close[key2])
    suit['dead'] = macd_df.close[key2] < macd_df.close[key1] and \
                   macd_df.dif[key2] > macd_df.dif[key1] > 0 and \
                   macd_df.macd[-2] > 0 > macd_df.macd[-1]
  except:
    pass
  return suit


# 获取行业涨幅
def get_industries_mom(calc_date, mom_days=4):
  start_date = get_trade_days(end_date=calc_date, count=mom_days)[0]
  ind_info_df = get_industries('sw_l1', date=start_date)

  ind_pct_df = finance.run_query(
    query(
      finance.SW1_DAILY_PRICE.code,
      finance.SW1_DAILY_PRICE.name,
      sum(finance.SW1_DAILY_PRICE.change_pct)
    ).filter(
      finance.SW1_DAILY_PRICE.code.in_(ind_info_df.index.tolist()),
      finance.SW1_DAILY_PRICE.date.between(start_date, calc_date)
    ).group_by(
      finance.SW1_DAILY_PRICE.code
    ).order_by(
      finance.SW1_DAILY_PRICE.change_pct.desc()
    )
  )
  print(ind_pct_df)
  return ind_pct_df


# 0-1 大盘顶背离清仓
def dapan(context):
  print(g.dbl)
  result = macd_gold_dead(g.wp_benchmark, context)['dead']
  g.dbl.append(result)
  g.dbl = g.dbl[-g.dbl_days:]
  if result:
    print('大盘顶背离，清仓')
    for stock in context.portfolio.positions.keys():
      custom_target_value(context, stock, 0)


def filter_basic_pool(end_date, inter_stocks=[]):
  # 过滤次新股（新股、老股的分界日期，两种指定方法）
  # 新老股的分界日期, 自然日180天
  # by_date = context.previous_date - datetime.timedelta(days=180)
  # 新老股的分界日期，120个交易日
  all_stocks = get_all_securities(date=get_trade_days(end_date=end_date, count=120)[0]).index.tolist()
  if len(inter_stocks) > 0:
    all_stocks = list(set(inter_stocks).intersection(set(all_stocks)))

  curr_data = get_current_data()
  return [stock for stock in all_stocks if not (
    stock.startswith(('68', '4', '8')) or  # 创业，科创，北交所
    curr_data[stock].paused or
    curr_data[stock].is_st or  # ST
    ('ST' in curr_data[stock].name) or
    ('*' in curr_data[stock].name) or
    ('退' in curr_data[stock].name))]


# 1-1 准备股票池
def prepare_stock_list(context):
  g.yesterday_HL_list = []
  # 获取已持有列表
  hold_list = list(context.portfolio.positions)
  if hold_list:
    g.yesterday_HL_list = get_price(
      hold_list, end_date=context.previous_date, frequency='daily',
      fields=['close', 'high_limit', 'paused'],
      count=1, panel=False).query('close == high_limit and paused == 0')['code'].tolist()
  # 判断今天是否为账户资金再平衡的日期
  # g.no_trading_today_signal = today_is_between(context)


# 1-2 选股模块
def get_stock_list(context):
  if len(g.stk_pool) == 0:
    prepare_stk_pool(context)

  curr_data = get_current_data()
  g.stk_pool = [stock for stock in g.stk_pool if not (
    (curr_data[stock].last_price == curr_data[stock].high_limit) or  # 涨停开盘, 其它时间用 last_price
    (curr_data[stock].last_price == curr_data[stock].low_limit)  # 跌停开盘, 其它时间用 last_price
  ) and (2 < curr_data[stock].last_price < 20)]

  # print('initial_list中含有{}个元素'.format(len(initial_list)))
  final_list = get_fundamentals(query(valuation.code,
                                      valuation.market_cap)
                                .filter(valuation.code.in_(g.stk_pool),
                                        valuation.market_cap > 5)
                                .order_by(valuation.market_cap.asc())
                                .limit(g.stock_num))['code'].tolist()

  log.info(f'今日前{g.stock_num}:{final_list}')
  return final_list


# 国九条：财务造假退市指标：==> 【C罗：按审计无保留意见过滤】
def filter_stocks_by_revenue_and_profit(context, stock_list):
  # 计算分红的三年起止时间
  time1 = context.previous_date
  time0 = time1 - datetime.timedelta(days=365 * 3)  # 三年
  # 计算年报的去年
  if time1.month >= 5:  # 5月后取去年
    last_year = str(time1.year - 1)
  else:  # 5月前取前年
    last_year = str(time1.year - 2)

  # print(f'按收入和盈利筛选前：{len(stock_list)}')
  # 2：主板亏损公司营业收入退市标准，组合指标修改为利润总额、净利润、扣非净利润三者孰低为负值，且营业收入低于 3 亿元。
  # get_history_fundamentals(security, fields, watch_date=None, stat_date=None, count=1, interval='1q', stat_by_year=False)
  list_len = len(stock_list)
  interval = 1000
  multiple_n = list_len // interval + 1
  start_i = 0
  stk_df = pd.DataFrame()
  for mul in range(multiple_n):
    start_i = mul * interval
    end_i = start_i + interval
    print(f'{start_i} - {end_i}')
    df = get_history_fundamentals(stock_list[start_i:end_i],
                                  fields=[income.operating_revenue, income.total_profit, income.net_profit],
                                  watch_date=None, stat_date=last_year, count=1, interval='1y', stat_by_year=True)
    # 扣非净利润找不到
    if len(stk_df) == 0:
      stk_df = df
    else:
      stk_df = pd.concat([stk_df, df])
  # 同时满足才剔除
  df = stk_df[(stk_df["operating_revenue"] < 3e8) & ((stk_df["total_profit"] < 0) | (stk_df["net_profit"] < 0))]
  bad_companies = list(df["code"])
  keep_list = [s for s in stock_list if s not in bad_companies]

  current_data = get_current_data()
  company_names_list = []
  company_names_ser = pd.Series(index=bad_companies)
  for item in bad_companies: company_names_ser[item] = current_data[item].name
  for item in bad_companies: company_names_list.append(current_data[item].name)
  df.loc[:, 'name'] = company_names_list
  df.set_index('code', inplace=True)

  print(f'剔除：营收太小 且 净利润为负的公司{len(company_names_ser)}个')
  return keep_list


# 基于开心果大妈策略改写
# 国九条：最近三个会计年度累计现金分红总额低于年均净利润的30%，且累计分红金额低于5000万元的，将被实施ST
def get_dividend_ratio_filter_list(context, stock_list, from_big_to_small=True, p1=0, p2=0.25):
  # 计算分红的三年起止时间
  time1 = context.previous_date
  time0 = time1 - datetime.timedelta(days=365 * 3)  # 三年
  # 计算年报的去年
  if time1.month >= 5:  # 5月后取去年
    last_year = str(time1.year - 1)
  else:  # 5月前取前年
    last_year = str(time1.year - 2)

  print(f'按分红筛选前：{len(stock_list)}')

  # 4：分红不达标ST：
  # 获取分红数据，由于finance.run_query最多返回4000行，以防未来数据超限，最好把stock_list拆分后查询再组合
  list_len = len(stock_list)
  interval = 1000
  multiple_n = list_len // interval + 1
  start_i = 0
  stk_df = pd.DataFrame()
  for mul in range(multiple_n):
    start_i = mul * interval
    end_i = start_i + interval
    print(f'{start_i} - {end_i}')
    # 截取不超过interval的列表并查询
    # STK_XR_XD 上市公司分红送股（除权除息）数据
    df = finance.run_query(query(
      finance.STK_XR_XD.code,
      finance.STK_XR_XD.a_registration_date,  # A股股权登记日
      finance.STK_XR_XD.bonus_amount_rmb  # 派息金额(人民币)万元 (现金分红)
    ).filter(
      finance.STK_XR_XD.a_registration_date >= time0,
      finance.STK_XR_XD.a_registration_date <= time1,
      finance.STK_XR_XD.code.in_(stock_list[start_i:end_i])))
    if len(stk_df) == 0:
      stk_df = df
    else:
      stk_df = pd.concat([stk_df, df])

  dividend = stk_df.fillna(0)
  dividend = dividend.set_index('code')
  dividend = dividend.groupby('code').sum()
  temp_list = list(dividend.index)  # query查询不到无分红信息的股票，所以temp_list长度会小于stock_list

  # 获取3年净利润数据
  # get_history_fundamentals(security, fields, watch_date=None, stat_date=None, count=1, interval='1q', stat_by_year=False)
  np = get_history_fundamentals(temp_list, fields=[income.net_profit], watch_date=None,
                                stat_date=last_year, interval='1y', count=3)
  # print("获取3年净利润数据:")
  # print(np.head(10))
  np = np.set_index('code')
  np = np.groupby('code').mean()

  # 获取市值相关数据，用于计算股息率
  q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(temp_list))
  cap = get_fundamentals(q, date=time1)
  cap = cap.set_index('code')

  # 沪深主板公司，最近三年现金分红低于年均净利润30%+三年累计分红少于5000万将被实施ST
  # 筛选 过去三年累计分红大于平均净利润的30% 或 累计分红>5000万
  DR = pd.concat([dividend, np, cap], axis=1, sort=True)
  df.set_index('code', inplace=True)
  # DR=DR[((DR['bonus_amount_rmb']*10000)>(DR['net_profit']*0.3)) & (DR['bonus_amount_rmb']>5000)]
  # C罗改：要保留的是 或。
  DR = DR[((DR['bonus_amount_rmb'] * 10000) >= (DR['net_profit'] * 0.3)) | (DR['bonus_amount_rmb'] >= 5000)]
  print(f'按付息5000万或30%以上筛选后的股票数量：{len(list(DR.index))}')

  # 计算股息率并筛选
  # DR['dividend_ratio'] = (DR['bonus_amount_rmb']/10000) / DR['market_cap']
  # 按股息率从大到小排序，ascending的意思时从小到大，所以取反
  # DR = DR.sort_values(by=['dividend_ratio'], ascending=not(from_big_to_small))
  # 按股息率筛选p1-p2区间的
  # final_list = list(DR.index)[int(p1*len(DR)):int(p2*len(DR))]

  # C罗改：暂时直接返回，不考虑股息率
  final_list = list(DR.index)

  return final_list


######################################################################################
# 2.1 筛选审计意见
def filter_audit(context, code):
  # C罗 filter_audit 有问题, 参见修改版
  # report_type只有0和1，分别是财务报表审计和内部控制审计
  # opinion_type_id，才是有用的

  # 获取审计意见，近三年内如果有不合格(report_type为2、3、4、5)的审计意见则返回False，否则返回True
  lstd = context.previous_date
  last_year = (lstd.replace(year=lstd.year - 3, month=1, day=1)).strftime('%Y-%m-%d')
  q = query(finance.STK_AUDIT_OPINION).filter(finance.STK_AUDIT_OPINION.code == code,
                                              finance.STK_AUDIT_OPINION.pub_date >= last_year)
  df = finance.run_query(q)
  df['report_type'] = df['report_type'].astype(str)
  contains_nums = df['report_type'].str.contains(r'2|3|4|5')
  return not contains_nums.any()


# 2.1 筛选审计意见：蒋老师提供
def filter_stocks_by_auditor_opinion_jqz(context, stock_list):
  print(f'按审计无保留意见筛选前：{len(stock_list)}')
  # type:(context,list)-> list
  # 剔除近三年内有不合格(opinion_type_id >2 且不是 6)审计意见的股票
  start_date = datetime.date(context.current_dt.year - 3, 1, 1).strftime('%Y-%m-%d')
  end_date = context.previous_date.strftime('%Y-%m-%d')
  df = finance.run_query(query(finance.STK_AUDIT_OPINION).filter(finance.STK_AUDIT_OPINION.code.in_(stock_list),
                                                                 finance.STK_AUDIT_OPINION.report_type == 0,
                                                                 # 0:财务报表审计报告
                                                                 finance.STK_AUDIT_OPINION.opinion_type_id > 2,
                                                                 # 1:无保留,2:无保留带解释性说明
                                                                 finance.STK_AUDIT_OPINION.opinion_type_id != 6,
                                                                 # 6:未经审计，季报
                                                                 finance.STK_AUDIT_OPINION.end_date >= start_date,
                                                                 finance.STK_AUDIT_OPINION.pub_date <= end_date))
  bad_companies = df['code'].unique().tolist()
  keep_list = [s for s in stock_list if s not in bad_companies]

  print(f'按审计无保留意意见筛选后：{len(keep_list)}')
  return keep_list


# 1-3 整体调整持仓
def adjust_position(context):
  # 如果中小综指MACD顶背离，则清仓20个交易日
  if True not in g.dbl[-g.dbl_days:]:
    trade_signal = today_trade_signal(context)
    if trade_signal is None:
      # 获取应买入列表
      g.target_list = get_stock_list(context)

      target_list = g.target_list[:g.stock_num]
      log.info(str(target_list))

      # 调仓卖出
      hold_list = list(context.portfolio.positions)
      for stock in hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
          close_position(context, stock)
        else:
          log.info("已持有[%s]" % (stock))
      # 调仓买入
      buy_security(context, target_list)


# 1-4 调整昨日涨停股票
def check_limit_up(context):
  now_time = context.current_dt
  if len(g.yesterday_HL_list) != 0:
    # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
    for stock in g.yesterday_HL_list:
      current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                               skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
      if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
        log.info(f"{stock} 涨停打开，卖出")
        position = context.portfolio.positions[stock]
        close_position(context, stock)
        g.reason_to_sell = 'limitup'
      else:
        log.info(f"{stock} 涨停，继续持有")


# 1-5 如果昨天有股票卖出或者买入失败，剩余的金额今天早上买入
def check_remain_amount(context):
  if g.reason_to_sell is 'limitup':  # 判断提前售出原因，如果是涨停售出则次日再次交易，如果是止损售出则不交易
    hold_list = list(context.portfolio.positions.keys())
    if len(hold_list) < g.stock_num:
      target_list = g.target_list
      # 剔除本周一曾买入的股票，不再买入
      # target_list = filter_not_buy_again(target_list)
      target_list = [stk for stk in target_list if stk not in list(context.portfolio.positions.keys())]
      target_list = target_list[:min(g.stock_num, len(target_list))]
      log.info('有余额可用' + str(round((context.portfolio.cash), 2)) + '元。' + str(target_list))
      buy_security(context, target_list)
    g.reason_to_sell = ''
  else:
    log.info('虽然有余额可用，但是为止损后余额，下周再交易')
    g.reason_to_sell = ''


# 1-6 下午检查交易
def check_afternoon(context):
  trade_signal = today_trade_signal(context)
  if trade_signal is None:
    check_morning(context)
    check_limit_up(context)
    check_high_volume(context)
    check_remain_amount(context)


# 1-7 早盘止盈止损
def check_morning(context):
  stock_df = get_price(security=get_index_stocks(g.wp_benchmark), end_date=context.previous_date,
                       frequency='daily', fields=['close', 'open'], count=1, panel=False)
  # down_ratio = abs((stock_df['close'] / stock_df['open'] - 1).mean())
  down_ratio = (stock_df['close'] / stock_df['open']).mean()
  if down_ratio <= g.stoploss_market:
    g.reason_to_sell = 'stoploss'
    log.debug("大盘惨跌,平均降幅{:.2%}".format(down_ratio))
    for stock in context.portfolio.positions.keys():
      custom_target_value(context, stock, 0)
  else:
    for stk in context.portfolio.positions:
      if context.portfolio.positions[stk].price / context.portfolio.positions[stk].avg_cost < g.stoploss_limit:
        custom_target_value(context, stk, 0)
        log.debug(f"收益止损,卖出{stk}")
        g.reason_to_sell = 'stoploss'
        continue

      result = macd_gold_dead(stk, context)['dead']
      if result:
        custom_target_value(context, stk, 0)
        log.debug(f"股票macd死叉,卖出{stk}")
        g.reason_to_sell = 'stoploss'


# 3-2 调整放量股票
def check_high_volume(context):
  current_data = get_current_data()
  for stock in context.portfolio.positions:
    if current_data[stock].paused or context.portfolio.positions[stock].closeable_amount == 0:
      continue
    if current_data[stock].last_price == current_data[stock].high_limit:
      continue

    stock_df = get_bars(stock, count=g.HV_duration, unit='1d', fields=['volume'], include_now=True, df=False)
    vol_ratio = stock_df['volume'][-1] / stock_df['volume'].mean()
    if (current_data[stock].last_price < current_data[stock].high_limit) and vol_ratio >= g.HV_ratio:
      log.info(f"{stock} 放量 {vol_ratio} 未涨停，卖出")
      close_position(context, stock)


# 3-1 交易模块-自定义下单
# def order_target_value_(security, value):
#     if value == 0:
#         pass
#         # log.debug("Selling out %s" % (security))
#     else:
#         log.debug("Order %s to value %f" % (security, value))
#     return order_target_value(security, value)


# 3-2 交易模块-开仓
def open_position(context, security, value):
  order = custom_target_value(context, security, value)
  return order != None and order.filled > 0


# 3-3 交易模块-平仓
def close_position(context, stk):
  order = custom_target_value(context, stk, 0)  # 可能会因停牌失败
  if order != None:
    return order.status == OrderStatus.held and order.filled == order.amount
  return False


# 3-4 买入模块
def buy_security(context, target_list):
  # 调仓买入
  position_count = len(context.portfolio.positions)
  target_num = len(target_list)
  if target_num > position_count:
    value = context.portfolio.cash / (target_num - position_count)
    for stock in target_list:
      if context.portfolio.positions[stock].total_amount == 0:
        # if stock not in context.portfolio.positions:
        if open_position(context, stock, value):
          log.info("买入[%s]（%s元）" % (stock, value))
          if len(context.portfolio.positions) == target_num:
            break


def calc_macd(context, code, short=9, long=26, dea=12):
  # 获取过去一段时间的收盘价数据，这里使用 context.previous_date 避免未来函数
  macd_df = get_price(code, end_date=context.previous_date, count=(short + long + dea) * 5, fields='close').dropna()
  # 将实时价格添加到收盘价数据中
  # macd_df.loc[context.current_dt] = get_current_data()[code].last_price

  macd_df['ema_short'] = macd_df['close'].ewm(span=short, adjust=False).mean()
  macd_df['ema_long'] = macd_df['close'].ewm(span=long, adjust=False).mean()
  macd_df['dif'] = macd_df['ema_short'] - macd_df['ema_long']
  macd_df['dea'] = macd_df['dif'].ewm(span=dea, adjust=False).mean()
  macd_df['macd'] = (macd_df['dif'] - macd_df['dea']) * 2

  del macd_df['ema_short'], macd_df['ema_long']
  return macd_df


# 4-1 判断当天是否可以交易 None为可以交易
def today_trade_signal(context):
  curr_date = context.current_dt.strftime('%m-%d')

  if (('01-05' <= curr_date <= '01-31')) or ('04-05' <= curr_date <= '04-30'):
    return 'MONTH_EMPTY'
    refer_code = '399303.XSHE'
    # # closes = history(30, '1d', ['close'], refer_code, df=False, skip_paused=True)['close']
    # closes = \
    #     history(30, unit='1d', field='close', security_list=[refer_code], df=True, skip_paused=False, fq='pre')[
    #         refer_code]
    # currect_price = get_current_data()[refer_code].day_open
    # np.append(closes, currect_price)
    # bbi_data = (closes[-3:].mean() + closes[-6:].mean() + closes[-12:].mean() + closes[-24:].mean()) / 4.0
    # return currect_price >= bbi_data

    macd_df = calc_macd(context, g.wp_benchmark)
    macd_df['diver'] = macd_df['macd'] - macd_df['macd'].shift(1)
    return macd_df['diver'][-1] > 0
  else:
    return None


# 4-2 清仓后次日资金可转
def close_account(context):
  trade_signal = today_trade_signal(context)
  if trade_signal:
    hold_list = list(context.portfolio.positions)
    if len(hold_list) != 0:
      for stock in hold_list:
        close_position(context, stock)


# 3-1 交易模块-自定义下单
def custom_target_value(context, security, value):
  stk_name = get_security_info(security, context.previous_date).display_name
  log.info(f"时间：{datetime.datetime.now()} {'卖出' if value == 0 else '买入'}：{security}，{stk_name} {value}")
  return order_target_value(security, value)
