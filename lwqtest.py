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
    )]

new_list = [list(factor_list[0])]
print(new_list)

for factor_list, coef_list in new_list:
    print(factor_list, coef_list)

for i in range(13):
    print(i)