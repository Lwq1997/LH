import easytrader

# >pip install  easytrader   -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# >pip install  pypiwin32   -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
user = easytrader.use('universal_client')

# user.connect(r'C:\同花顺软件\同花顺\xiadan.exe') # 公司win
user.connect(r'F:\soft\同花顺\xiadan.exe')  # 家庭win

print("获取账户信息", user.balance)
print("获取持仓情况", user.position)
print(user.buy('600255', price=0, amount=100))
print(user.sell('600255', price=0, amount=100))

# target = 'jq'  # joinquant
# follower = easytrader.follower(target)
# follower.login(user='18291880968', password='19970820Lwq.')
# url = 'https://www.joinquant.com/algorithm/live/index?backtestId=75bb5b6fc27373410da1d23932e984a1'
# follower.follow(user, url, trade_cmd_expire_seconds=100000000000, cmd_cache=False)