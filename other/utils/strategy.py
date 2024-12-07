def run_strategy(current_price, price_prediction, actual_3_day_returns):
    total_returns = 0
    total_positive_return_count = 0
    total_negative_return_count = 0
    returns = []

    test = 0

    for i in range(len(price_prediction)):
        if price_prediction[i][2] > current_price[i][2]:
            total_returns += actual_3_day_returns[i]
            test += 100 * actual_3_day_returns[i]
            returns.append(actual_3_day_returns[i])
            if returns[-1] > 0:
                total_positive_return_count += 1
            else:
                total_negative_return_count += 1
        else:
            returns.append(0)

    success_rate = total_positive_return_count / (
        total_positive_return_count + total_negative_return_count
    )

    return {
        "total_returns": total_returns,
        "success_rate": success_rate,
        "total_positive_return_count": total_positive_return_count,
        "total_negative_return_count": total_negative_return_count,
        "returns": returns,
        "test": test,
    }
