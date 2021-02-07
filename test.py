def compute_metrics_fn(logits, target):
    id2label = ['整体满意度', '服务态度', '维修/作业品质', '环境整洁程度', '到店便利性', '其他', '洗车质量', '无意义',
                '作业时间的合理性', '维修保养过程的透明性', '活动及优惠', '消费体验', '跟踪/回访', '迅速出迎接待',
                '服务顾问交车专业性', '服务顾问接车专业性', '诚信', '预约接待', '休息区设施的舒适性', '餐饮服务', '材料费用合理性',
                '工时费用合理性', '客服等其他服务人员服务质量', '交车过程的效率', '保养/维修后的车辆状况', '承诺时间内完成', '经销商其他服务']
    threshhold = 0.3
    labels = target
    preds = (logits > threshhold) + 0
    f1_total = 0
    for i in range(labels.shape[1]):
        right_num = 0
        label_num = 1e-10
        pred_num = 1e-10

        for l, pr in zip(labels[:, i], preds[:, i]):
            if l == pr == 1:
                right_num += 1
            if l == 1:
                label_num += 1
            if pr == 1:
                pred_num += 1
        precision = round(right_num / pred_num, 2)
        recall = round(right_num / label_num, 2)
        f1 = round(2 * precision * recall / (precision + recall + 1e-10), 2)
        f1_total += f1
        print({id2label[i]: {'precision': precision, 'recall': recall, 'f1': f1}})

    return {'avg_f1': round(f1_total / labels.shape[1], 2)}