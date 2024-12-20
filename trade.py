import easytrader

user = easytrader.use('universal_client')

user.connect(r'C:\同花顺软件\同花顺\xiadan.exe') # 类似 r'C:\htzqzyb2\xiadan.exe'


print("获取账户信息",user.balance)
print("获取持仓情况",user.position)

target = 'jq'  # joinquant
follower = easytrader.follower(target)
follower.login(user='18291880968', password='19970820Lwq.')
