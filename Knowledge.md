
Multivariate Linear Regression:指的是同时拟合多个输出值
Multiple Linear Regression:多输入拟合

关闭科学计数法显示:np.set_printoptions(suppress=True)

AUC评估
# AUC: x:FP, y:TP
# AUC = 1，是完美分类器。
# AUC = [0.85, 0.95], 效果很好
# AUC = [0.7, 0.85], 效果一般
# AUC = [0.5, 0.7],效果较低，但用于预测股票已经很不错了
# AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。
# AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。
