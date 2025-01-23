# 克隆自聚宽文章：https://www.joinquant.com/post/51885
# 标题：单账户多策略架构
# 作者：深来浅止

from datetime import timedelta;
from jqdata import *;


# ======================JoinQuant.py-start======================
# 推荐在[研究环境]下创建"JoinQuant.py"文件

class JoinQuant:
    '''聚宽交易类封装'''

    def __init__(self):
        self.isDebug = True;
        self.trader = Trader();
        pass

    def Calibration(nowCash: float, cash: float):
        '''
        校准现金和股票持仓
        '''
        inout_cash(cash - nowCash);

        pass

    def Buy(self, security: str, price: float, amount: int):
        amount = self.RoundDownToNearest100(amount);
        if amount < 100:
            return None;
        result = order(security, amount, LimitOrderStyle(price));
        if not (result is None) and self.isDebug == False:
            self.trader.Buy(security.split('.')[0], price, amount);
        return result;

    def Sell(self, security: str, price: float, amount: int):
        amount = self.RoundDownToNearest100(amount);
        if amount == 0:
            return None;
        result = order(security, -amount, LimitOrderStyle(price));
        if not (result is None) and self.isDebug == False:
            self.trader.Sell(security.split('.')[0], price, amount);
        return result;

    def MarketBuy(self, security: str, amount: int):
        amount = self.RoundDownToNearest100(amount);
        if amount < 100:
            return None;
        result = order(security, amount);
        if not (result is None) and self.isDebug == False:
            self.trader.MarketBuy(security.split('.')[0], amount);
        return result;

    def MarketSell(self, security: str, amount: int):
        amount = self.RoundDownToNearest100(amount);
        if amount == 0:
            return None;
        result = order(security, -amount);
        if not (result is None) and self.isDebug == False:
            self.trader.MarketSell(security.split('.')[0], amount);
        return result;

    def MoneyBuy(self, money: float, security: str, price: float):
        amount = self.RoundDownToNearest100(money // price);
        if amount < 100:
            return None;
        return self.Buy(security, price, amount);

    def MoneySell(self, money: float, security: str, price: float):
        amount = self.RoundDownToNearest100(money // price);
        if amount == 0:
            return None;
        return self.Sell(security, price, amount);

    def AdjustHoldValue(self, cash: float, money: float, security: str, nowPrice: float, nowAmount: int,
                        isMarket: bool = True):
        '''
        调整持有价值
        多了就卖出，少了就买入
        Args:
            cash: 持有现金
            money: 想要调整到多少
            security: 股票码
            nowPrice: 当前股票价格
            nowAmount: 当前持有数量
            isMarket: 是否按市价操作
        '''
        nowMoney = nowPrice * nowAmount;
        # 要买入
        if nowMoney < money:
            differenceMoney = money - nowMoney;
            amount = self.RoundDownToNearest100(differenceMoney // nowPrice);
            if amount < 100 or cash < differenceMoney:
                return None;
            if isMarket == True:
                return self.MarketBuy(security, amount);
            else:
                return self.Buy(security, nowPrice, amount);
        # 要卖出
        else:
            differenceMoney = nowMoney - money;
            amount = self.RoundDownToNearest100(differenceMoney // nowPrice);
            if amount < 100:
                return None;
            if isMarket == True:
                return self.MarketSell(security, amount);
            else:
                return self.Sell(security, nowPrice, amount);

    def IsMultipleOf100(self, n):
        return n % 100 == 0;

    def RoundDownToNearest100(self, n):
        return int((n // 100) * 100);


# 和外部通讯类，此处可以实现你自己的交易信号。例如调用服务器接口。
class Trader:
    def __init__(self):
        self.token = '';
        self.headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token};

    def SetToken(self, token: str):
        self.token = token;
        self.headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token};

    def Buy(self, stockNumber: str, price: float, amount: int):
        pass;

    def Sell(self, stockNumber: str, price: float, amount: int):
        pass;

    def MarketBuy(self, stockNumber: str, amount: int):
        pass;

    def MarketSell(self, stockNumber: str, amount: int):
        pass;


# ======================JoinQuant.py-end======================

# ======================Combin.py-start======================
# 推荐在[研究环境]下创建"Combin.py"文件
# 多策略核心相关类

class Schedule:
    '''定时任务类'''

    def __init__(self, period: str = '', func: str = '', force: bool = False, monthday: int = 0, weekday: int = 0,
                 time: str = '', reference_security: str = ''):
        self.period: str = period;
        self.func: str = func;
        self.force: bool = force;
        self.monthday: int = monthday;
        self.weekday: int = weekday;
        self.time: str = time;
        self.reference_security: str = reference_security;


class ManageStrategy:
    '''管理类'''

    def __init__(self):
        self.context = {};
        # 策略字典 键是策略唯一标识
        self.strategyMap: dict[int, AStrategy] = {};
        # 可用现金
        self.availableCash = 0;
        pass

    def PositionDayUpdate(self):
        '''每日更新所有策略的标的仓位，解锁挂单和可卖出仓位'''
        for key in self.strategyMap:
            removeList = [];
            for position in self.strategyMap[key].holdList:
                position.closeableAmount = position.totalAmount;
                position.todayAmount = 0;
                position.lockedAmount = 0;
                if position.totalAmount == 0:
                    removeList.append(position);
                    self.strategyMap[key].holdMap.pop(position.security);

            for item in removeList:
                self.strategyMap[key].holdList.remove(item);

            removeList.clear();

    def Buy(self, strategyID: int, security: str, price: float, amount: int):
        if strategyID not in self.strategyMap:
            return None;

        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.Buy(security, price, amount);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, True, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def Sell(self, strategyID: int, security: str, price: float, amount: int):
        if strategyID not in self.strategyMap:
            return None;

        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.Sell(security, price, amount);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, False, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def MarketBuy(self, strategyID: int, security: str, amount: int):
        if strategyID not in self.strategyMap:
            return None;

        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.MarketBuy(security, amount);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, True, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def MarketSell(self, strategyID: int, security: str, amount: int):
        if strategyID not in self.strategyMap:
            return None;

        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.MarketSell(security, amount);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, False, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def MoneyBuy(self, strategyID: int, money: float, security: str, price: float):
        if strategyID not in self.strategyMap:
            return None;

        price = round(price, 3);
        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.MoneyBuy(money, security, price);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, True, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def MoneySell(self, strategyID: int, money: float, security: str, price: float):
        if strategyID not in self.strategyMap:
            return None;

        price = round(price, 3);
        strategy = self.strategyMap[strategyID];
        result = strategy.joinQuant.MoneySell(money, security, price);
        if result is None:
            return None;

        self.UpdatePosition(strategy, security, False, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def AdjustHoldValue(self, strategyID: int, money: float, security: str, nowPrice: float, isMarket: bool = True):
        '''
        调整持有价值
        多了就卖出，少了就买入

        Args:
            strategyID: 策略唯一标识
            money: 想要调整到多少
            security: 股票码
            nowPrice: 当前股票价格
            isMarket: 是否按市价操作
        '''

        if strategyID not in self.strategyMap:
            return None;

        nowPrice = round(nowPrice, 3);

        strategy = self.strategyMap[strategyID];
        nowAmount = 0;

        if security in strategy.holdMap:
            nowAmount = strategy.holdMap[security].totalAmount + strategy.holdMap[security].lockedAmount;

        result = strategy.joinQuant.AdjustHoldValue(strategy.availableCash, money, security, nowPrice, nowAmount,
                                                    isMarket);
        if result is None:
            return None;

        nowMoney = nowPrice * nowAmount;
        isBuy = False;
        # 要买入
        if nowMoney < money:
            isBuy = True;
        self.UpdatePosition(strategy, security, isBuy, result.filled, result.price, result.commission, result.add_time);
        # 每次交易完成后需要更新现金分配
        self.AllocationCash();
        return result;

    def UpdatePosition(self, strategy: 'AStrategy', security: str, isBuy: bool, amount: int, price: float,
                       commission: float, time: datetime.datetime):
        '''
        更新策略中标的信息

        Args:
            strategy: 策略对象
            security: 标
            isBuy: 是否购买
            amount: 数量
            price: 单价
            commission: 交易费用（佣金、税费等）
            time: 时间
        '''
        if isBuy == True:
            # 使用以下公式计算出来的现金不准，原因不明
            # strategy.availableCash = strategy.availableCash - amount * price - commission;
            # 第一次购买需要创建新对象
            if security not in strategy.holdMap:
                strategy.holdMap[security] = EPosition(security);
                strategy.holdMap[security].security = security;
                strategy.holdMap[security].initTime = time;
                strategy.holdList.append(strategy.holdMap[security]);

            strategy.holdMap[security].todayAmount = amount;
            strategy.holdMap[security].totalAmount = strategy.holdMap[security].totalAmount + amount;
            strategy.holdMap[security].transactTime = time;
        else:
            # strategy.availableCash = strategy.availableCash + amount * price - commission;
            strategy.holdMap[security].closeableAmount = strategy.holdMap[security].closeableAmount - amount;
            strategy.holdMap[security].totalAmount = strategy.holdMap[security].totalAmount - amount;
            strategy.holdMap[security].transactTime = time;
            # 如果持有量为0，则从持有列表中移除
            if strategy.holdMap[security].totalAmount == 0:
                strategy.holdList.remove(strategy.holdMap[security]);
                strategy.holdMap.pop(security);

    def AllocationCash(self):
        '''根据权重分配现金'''
        self.availableCash = self.context.portfolio.available_cash;
        # 权重总和
        weightTotal = 0;
        # 总现金
        cashTotal = self.availableCash;
        # 分配后剩余现金
        remainingCash = 0;
        # 权重最大的策略
        heaviestKey = '';

        tempWeight = 0;
        for key in self.strategyMap:
            # 使用大于等于，防止只有一个策略时heaviestKey为空
            if self.strategyMap[key].weight >= tempWeight:
                tempWeight = self.strategyMap[key].weight;
                heaviestKey = key;
            weightTotal = weightTotal + self.strategyMap[key].weight;

        if weightTotal == 0:
            return;

        remainingCash = cashTotal;
        for key in self.strategyMap:
            self.strategyMap[key].availableCash = math.floor(cashTotal * (self.strategyMap[key].weight / weightTotal));
            remainingCash = remainingCash - self.strategyMap[key].availableCash;

        # 将剩余金额分配给权重最大的策略
        self.strategyMap[heaviestKey].availableCash = self.strategyMap[heaviestKey].availableCash + remainingCash;


class EPosition:
    '''标实体类'''

    def __init__(self, security: str):
        # 股票代码
        self.security = security;
        # 建仓时间
        self.initTime: datetime.datetime;
        # 最后交易时间
        self.transactTime: datetime.datetime;
        # 挂单冻结仓位
        self.lockedAmount = 0;
        #  总仓位, 但不包括挂单冻结仓位( 如果要获取当前持仓的仓位,需要将lockedAmount和totalAmount相加)
        self.totalAmount = 0;
        # 可卖出的仓位
        self.closeableAmount = 0;
        # 今天开的仓位
        self.todayAmount = 0;


class AStrategy:
    '''策略类抽象类'''

    def __init__(self, mStrategy: ManageStrategy):
        self.mStrategy = mStrategy;
        # 策略唯一标识
        self.id = 0;
        # 策略名称
        self.name = '';
        # 策略类名
        self.className = '';
        # 可用现金
        self.availableCash = 0;
        # 策略权重（现金分配占比）
        self.weight = 0;
        # 原始策略权重（部分策略有空窗期，结束空窗期时需要调回设定的权重）
        self.originalWeight = 0;
        # 买入时溢价金额比例，由于同花顺市价买入偶尔有问题，改用限价买入
        self.premiumRatio = 0.015;
        # 当前持有的标列表
        self.holdList: list[EPosition] = [];
        self.holdMap: dict[str, EPosition] = {};
        # 封装的交易函数
        self.joinQuant = JoinQuant();
        self.scheduleList: list[Schedule] = [];

    def Init():
        pass

    def InitJoinQuant(self, isDebug: bool, traderToken: str, traderTerminalGUID: str):
        '''
            初始化封装的交易类

            Args:
                isDebug: 是否Debug模式 debug模式下不会向服务器发送请求
                traderToken: 用户Token
                traderTerminalGUID: 交易终端GUID
        '''
        self.joinQuant.isDebug = isDebug;
        self.joinQuant.trader.strategyID = self.id;
        self.joinQuant.trader.SetToken(traderToken);
        self.joinQuant.trader.terminalGUID = traderTerminalGUID;
        pass

    def RegisterSchedule(self):
        '''注册策略执行函数'''
        pass

    def Log(self):
        '''日志'''
        pass


class Filter:
    '''过滤器'''

    @staticmethod
    def KCBJ(stockList: list):
        '''过滤科创北交'''
        for stock in stockList[:]:
            if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
                stockList.remove(stock);
        return stockList;

    @staticmethod
    def Paused(stockList: list):
        '''过滤停牌股票'''
        current_data = get_current_data();
        return [stock for stock in stockList if not current_data[stock].paused];

    @staticmethod
    def ST(stockList: list):
        '''过滤ST及其他具有退市标签的股票'''
        current_data = get_current_data();
        return [stock for stock in stockList
                if not current_data[stock].is_st
                and 'ST' not in current_data[stock].name
                and '*' not in current_data[stock].name
                and '退' not in current_data[stock].name]

    @staticmethod
    def LimitUp(holdStockList: list, stockList: list, unit: str = '1m'):
        '''
        过滤涨停的股票

        Args:
            holdStockList: 持有的股票列表
            stockList: 需要过滤的股票列表
            unit: 单位时间长度, 几天或者几分钟, 现在支持'Xd','Xm', X是一个正整数, 分别表示X天和X分钟(不论是按天还是按分钟回测都能拿到这两种单位的数据), 注意, 当X > 1时, field只支持['open', 'close', 'high', 'low', 'volume', 'money']这几个标准字段.
        '''
        last_prices = history(1, unit=unit, field='close', security_list=stockList);
        current_data = get_current_data();
        # 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
        return [stock for stock in stockList if
                stock in holdStockList or last_prices[stock][-1] < current_data[stock].high_limit];

    @staticmethod
    def LimitDown(holdStockList: list, stockList: list, unit: str = '1m'):
        '''
            过滤跌停的股票

            Args:
                holdStockList: 持有的股票列表
                stockList: 需要过滤的股票列表
                unit: 单位时间长度, 几天或者几分钟, 现在支持'Xd','Xm', X是一个正整数, 分别表示X天和X分钟(不论是按天还是按分钟回测都能拿到这两种单位的数据), 注意, 当X > 1时, field只支持['open', 'close', 'high', 'low', 'volume', 'money']这几个标准字段.
        '''
        last_prices = history(1, unit=unit, field='close', security_list=stockList);
        current_data = get_current_data();

        return [stock for stock in stockList if
                stock in holdStockList or last_prices[stock][-1] > current_data[stock].low_limit];

    @staticmethod
    def NewStock(previousDate: datetime.date, days: float, stockList: list):
        '''
        过滤次新股

        Args:
            previousDate: 前一个交易日
            days: 上市天数
            stockList: 需要过滤的股票列表
        '''
        return [stock for stock in stockList if
                not previousDate - get_security_info(stock).start_date < datetime.timedelta(days=days)];

    @staticmethod
    def LockedShares(previousDate: datetime.date, days: float, stockList: list, rate: float = 0.2):
        '''
        过滤大幅解禁

        Args:
            previousDate: 前一个交易日
            days: 上市天数
            rate: 解禁占比
            stockList: 需要过滤的股票列表
        '''
        df = get_locked_shares(stock_list=stockList, start_date=previousDate.strftime('%Y-%m-%d'), forward_count=days);
        # 解禁数量占总股本的百分比
        df = df[df['rate1'] > rate];
        filterlist = list(df['code']);
        return [stock for stock in stockList if stock not in filterlist];

    @staticmethod
    def HighPrice(price: float, holdStockList: list, stockList: list, unit: str = '1m'):
        '''
            过滤股价高于price的股票

            Args:
                price: 股票价格
                holdStockList: 持有的股票列表
                stockList: 需要过滤的股票列表
                unit: 单位时间长度, 几天或者几分钟, 现在支持'Xd','Xm', X是一个正整数, 分别表示X天和X分钟(不论是按天还是按分钟回测都能拿到这两种单位的数据), 注意, 当X > 1时, field只支持['open', 'close', 'high', 'low', 'volume', 'money']这几个标准字段.
        '''
        last_prices = history(1, unit=unit, field='close', security_list=stockList);
        return [stock for stock in stockList if stock in holdStockList or last_prices[stock][-1] < price];

    @staticmethod
    def GetDividendRatioList(previousDate: datetime.date, stockList: list, sort, p1, p2):
        '''
            根据最近一年分红除以当前总市值计算股息率并筛选

            Args:
                previousDate: 昨天的日期
                stockList: 需要过滤的股票列表
                sort: 是否顺序排序
                p1: 小值
                p2: 大值
        '''
        time1 = previousDate;
        time0 = time1 - datetime.timedelta(days=365);
        # 获取分红数据，由于finance.run_query最多返回4000行，以防未来数据超限，最好把stock_list拆分后查询再组合
        # 某只股票可能一年内多次分红，导致其所占行数大于1，所以interval不要取满4000
        interval = 1000;
        list_len = len(stockList);
        # 截取不超过interval的列表并查询
        q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.a_registration_date,
                  finance.STK_XR_XD.bonus_amount_rmb).filter(
            finance.STK_XR_XD.a_registration_date >= time0,
            finance.STK_XR_XD.a_registration_date <= time1,
            finance.STK_XR_XD.code.in_(stockList[:min(list_len, interval)]));
        df = finance.run_query(q);
        # 对interval的部分分别查询并拼接
        if list_len > interval:
            df_num = list_len // interval;
            for i in range(df_num):
                q = query(finance.STK_XR_XD.code, finance.STK_XR_XD.a_registration_date,
                          finance.STK_XR_XD.bonus_amount_rmb).filter(
                    finance.STK_XR_XD.a_registration_date >= time0,
                    finance.STK_XR_XD.a_registration_date <= time1,
                    finance.STK_XR_XD.code.in_(stockList[interval * (i + 1):min(list_len, interval * (i + 2))]))
                temp_df = finance.run_query(q);
                df = df.append(temp_df);
        dividend = df.fillna(0);
        dividend = dividend.set_index('code');
        dividend = dividend.groupby('code').sum();
        # query查询不到无分红信息的股票，所以temp_list长度会小于stock_list
        temp_list = list(dividend.index);
        # 获取市值相关数据
        q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(temp_list));
        cap = get_fundamentals(q, date=time1);
        cap = cap.set_index('code');
        # 计算股息率
        DR = pd.concat([dividend, cap], axis=1, sort=False);
        DR['dividend_ratio'] = (DR['bonus_amount_rmb'] / 10000) / DR['market_cap'];
        # 排序并筛选
        DR = DR.sort_values(by=['dividend_ratio'], ascending=sort);
        final_list = list(DR.index)[int(p1 * len(DR)): int(p2 * len(DR))];
        return final_list;


# ======================Combin.py-end======================


# =======================================================================================
# 以下为正常使用时策略的代码（假设已经在[研究环境]下添加了"JoinQuant.py"和"Combin.py"）
# =======================================================================================
def initialize(context):
    log.set_level('order', 'error');
    set_option('use_real_price', True);
    set_option('avoid_future_data', True);
    set_benchmark('000905.XSHG');
    # 设置交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=2.5 / 10000, close_commission=2.5 / 10000,
                             close_today_commission=0, min_commission=1), type='stock')
    set_order_cost(
        OrderCost(open_tax=0, close_tax=0, open_commission=0.0002, close_commission=0.0002, close_today_commission=0,
                  min_commission=1), type='fund')

    # ================注册策略-start================
    mStrategy = ManageStrategy();
    mStrategy.context = context;
    mStrategy.availableCash = context.portfolio.available_cash;
    g.MStrategy = mStrategy;

    startegyZHQDM = StartegyZHQDM(mStrategy);
    startegyZHQDM.id = 1;
    startegyZHQDM.InitJoinQuant(True, '', '');
    mStrategy.strategyMap[startegyZHQDM.id] = startegyZHQDM;

    startegyB = StartegyB(mStrategy);
    startegyB.id = 2;
    startegyB.InitJoinQuant(True, '', '');
    mStrategy.strategyMap[startegyB.id] = startegyB;

    mStrategy.AllocationCash();
    RegisterSchedule();
    # ================注册策略-end================


def RegisterSchedule():
    '''注册策略的定期执行方法'''
    # 9:00时各个策略会执行CheckCanTradeDay进行权重调整
    run_daily(Execute_PositionDayUpdate, time='9:01', reference_security='000300.XSHG');

    for key in g.MStrategy.strategyMap:
        strategy = g.MStrategy.strategyMap[key];
        scheduleList = strategy.scheduleList;
        # 没有设置定时任务则获取策略默认定时任务
        if len(scheduleList) == 0:
            scheduleList = strategy.RegisterSchedule();
        RegisterRun(scheduleList);


def RegisterRun(scheduleList: list):
    '''注册run_xxx方法'''
    for schedule in scheduleList:
        if schedule.period == "daily":
            run_daily(globals().get(schedule.func), time=schedule.time)
        elif schedule.period == "weekly":
            run_weekly(globals().get(schedule.func), schedule.weekday, time=schedule.time)
        elif schedule.period == "monthly":
            run_monthly(globals().get(schedule.func), schedule.monthday, time=schedule.time)


def Execute_PositionDayUpdate(context):
    '''更新标'''
    g.MStrategy.context = context;
    g.MStrategy.availableCash = context.portfolio.available_cash;
    g.MStrategy.PositionDayUpdate();
    g.MStrategy.AllocationCash();


# ===================StartegyZHQDM_1-start===================
def Execute_StartegyZHQDM_1_CheckCanTradeDay(context):
    g.MStrategy.strategyMap[1].CheckCanTradeDay(context.current_dt);


def Execute_StartegyZHQDM_1_GetHighLimitStockList(context):
    g.MStrategy.strategyMap[1].GetHighLimitStockList(context.previous_date);


def Execute_StartegyZHQDM_1_Trader(context):
    g.MStrategy.strategyMap[1].Trader(context.previous_date);


def Execute_StartegyZHQDM_1_CheckLimitUp(context):
    g.MStrategy.strategyMap[1].CheckLimitUp();


def Execute_StartegyZHQDM_1_SellAll(context):
    g.MStrategy.strategyMap[1].SellAll();


# ===================StartegyZHQDM_1-end===================

# ===================StartegyB_2-start===================
def Execute_StartegyB_2_CheckCanTradeDay(context):
    g.MStrategy.strategyMap[2].CheckCanTradeDay(context.current_dt);


def Execute_StartegyB_2_Adjust(context):
    g.MStrategy.strategyMap[2].Adjust(context.current_dt);


def Execute_StartegyB_2_Test(context):
    g.MStrategy.strategyMap[2].Test();


def Execute_StartegyB_2_SellAll(context):
    g.MStrategy.strategyMap[2].SellAll();


# ===================StartegyB_2-end===================

# 策略A
class StartegyZHQDM(AStrategy):
    '''正黄旗大妈选股法'''

    def __init__(self, mStrategy):
        super().__init__(mStrategy);
        self.name = '正黄旗大妈选股法';
        self.className = 'StartegyZHQDM';
        self.originalWeight = 0.5;
        self.weight = self.originalWeight;
        # 当前持仓中，涨停股票列表
        self.highLimitList: list = [];
        # 最大持股数量
        self.maxStockCount = 6;
        # 最大价格
        self.maxPrice = 9;
        self.isCanTrade = False;

    def RegisterSchedule(self):
        return [
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_CheckCanTradeDay', time='9:00'),
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_GetHighLimitStockList', time='9:05'),
            Schedule(period='monthly', func=f'Execute_{self.className}_{self.id}_Trader', monthday=5, time='10:30'),
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_CheckLimitUp', time='14:00'),
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_SellAll', time='14:50')
        ];

    def CheckCanTradeDay(self, currentDate: datetime.date):
        '''判断是否能交易的日期'''
        today = currentDate.strftime('%m-%d');
        if (('01-01' <= today) and (today <= '01-31')) or (('04-01' <= today) and (today <= '04-30')):
            self.isCanTrade = False;
            self.weight = 0;
        else:
            self.isCanTrade = True;
            self.weight = self.originalWeight;

    def Trader(self, previousDate: datetime.date):
        dt_last = previousDate;
        stockList = get_all_securities('stock', dt_last).index.tolist();
        stockList = Filter.KCBJ(stockList);
        # 高股息(全市场最大25%)
        stockList = Filter.GetDividendRatioList(dt_last, stockList, False, 0, 0.25);

        q = query(valuation.code,
                  valuation.pe_ratio / indicator.inc_net_profit_year_on_year,  # PEG
                  indicator.roe / valuation.pb_ratio,  # PB-ROE
                  indicator.roe).filter(
            valuation.pe_ratio / indicator.inc_net_profit_year_on_year > -1,
            valuation.pe_ratio / indicator.inc_net_profit_year_on_year < 3,
            # indicator.roe / valuation.pb_ratio > 0,
            valuation.code.in_(stockList));
        df_fundamentals = get_fundamentals(q, date=None);
        stockList = list(df_fundamentals.code);
        # fuandamental data
        df = get_fundamentals(
            query(valuation.code).filter(valuation.code.in_(stockList)).order_by(valuation.market_cap.asc()));

        holdSecurityList = [stock.security for stock in self.holdList];
        choiceList = list(df.code);
        choiceList = Filter.ST(choiceList);
        choiceList = Filter.Paused(choiceList);
        choiceList = Filter.LimitUp(holdSecurityList, choiceList);
        choiceList = Filter.LimitDown(holdSecurityList, choiceList);
        choiceList = Filter.HighPrice(self.maxPrice, holdSecurityList, choiceList);
        choiceList = choiceList[:self.maxStockCount];
        cdata = get_current_data();

        # 卖出
        # 卖出时会对holdMap进行操作，因此需要拷贝一个字典进行循环
        newMap = self.holdMap.copy();
        for stock in newMap:
            if (stock not in choiceList):
                result = self.mStrategy.MarketSell(self.id, stock, self.holdMap[stock].closeableAmount);
                if not (result is None):
                    log.info('Sell', stock, cdata[stock].name);
        # 买入
        if self.maxStockCount > len(self.holdMap):
            for stock in choiceList:
                if stock not in self.holdMap:
                    psize = self.availableCash / (self.maxStockCount - len(self.holdMap));
                    result = self.mStrategy.AdjustHoldValue(self.id, psize, stock, cdata[stock].last_price);
                    if not (result is None):
                        log.info('buy', stock, cdata[stock].name);
                    if len(self.holdMap) == self.maxStockCount:
                        break;

    # 准备股票池
    def GetHighLimitStockList(self, previousDate: datetime.date):
        self.highLimitList = [];
        if len(self.holdList) > 0:
            securityList = [stock.security for stock in self.holdList];
            df = get_price(securityList, end_date=previousDate, frequency='daily', fields=['close', 'high_limit'],
                           count=1, panel=False);
            self.highLimitList = df[df['close'] == df['high_limit']]['code'].tolist();

    #  调整昨日涨停股票
    def CheckLimitUp(self):
        # 获取持仓的昨日涨停列表
        current_data = get_current_data();
        if len(self.highLimitList) > 0:
            for stock in self.highLimitList:
                if current_data[stock].last_price < current_data[stock].high_limit:
                    if stock in self.holdMap:
                        result = self.mStrategy.MarketSell(self.id, stock, self.holdMap[stock].closeableAmount);
                        if not (result is None):
                            log.info(f"{stock}涨停打开，卖出");
                else:
                    log.info(f"{stock}涨停，继续持有");

    def SellAll(self):
        '''在不进行交易的时间段清仓'''
        if self.isCanTrade == False:
            if len(self.holdList) == 0:
                return;
            print(f"=============={self.name} 空仓==============");
            newMap = self.holdMap.copy();
            for key in newMap:
                result = self.mStrategy.MarketSell(self.id, key, self.holdMap[key].closeableAmount);


# 策略B 1/4月买入纳指（仅仅只是演示用的策略）
class StartegyB(AStrategy):
    '''策略B'''

    def __init__(self, mStrategy):
        super().__init__(mStrategy);
        self.name = '策略B';
        self.className = 'StartegyB';
        self.originalWeight = 0.5;
        self.weight = self.originalWeight;
        self.premiumRatio = 0.008;
        self.isCanTrade = False;

    # 策略自有的run_xxx方法
    def RegisterSchedule(self):
        return [
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_CheckCanTradeDay', time='9:00'),
            Schedule(period='weekly', func=f'Execute_{self.className}_{self.id}_Adjust', weekday=2, time='10:00'),
            Schedule(period='weekly', func=f'Execute_{self.className}_{self.id}_Test', weekday=2, time='10:00'),
            Schedule(period='daily', func=f'Execute_{self.className}_{self.id}_SellAll', time='14:50'),
        ];

    def CheckCanTradeDay(self, currentDate: datetime.date):
        '''判断是否能交易的日期'''
        today = currentDate.strftime('%m-%d');
        if (('01-02' <= today) and (today <= '01-31')) or (('04-02' <= today) and (today <= '04-30')):
            self.isCanTrade = True;
            self.weight = self.originalWeight;
        else:
            self.isCanTrade = False;
            self.weight = 0;

    def Adjust(self, currentDate: datetime.date):
        '''调仓 没有持仓就买入，有持仓就卖出（仅演示多策略用，此购买逻辑是不对的）'''
        if self.isCanTrade == False or self.weight == 0:
            return;

        etf = '513100.XSHG';
        cdata = get_current_data();
        self.mStrategy.AdjustHoldValue(self.id, self.availableCash, etf,
                                       cdata[etf].last_price * (1 + self.premiumRatio), False);

    def Test(self):
        pass;

    def SellAll(self):
        '''在不进行交易的时间段清仓'''
        if self.isCanTrade == False:
            if len(self.holdList) == 0:
                return;
            print(f"=============={self.name} 空仓==============");
            newMap = self.holdMap.copy();
            for key in newMap:
                result = self.mStrategy.MarketSell(self.id, key, self.holdMap[key].closeableAmount);
