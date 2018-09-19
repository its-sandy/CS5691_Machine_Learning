function error = RMSE(y1, y2)

error = sqrt(mean((y1-y2).^2));

end