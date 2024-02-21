function data =  pre_process(data)

num = size(data,1);
mean_row = mean(data);
std_row = std(data);
mean_total = repmat(mean_row, num, 1);
std_row = repmat(std_row, num, 1);

data = (data - mean_total) ./ std_row;
data(isnan(data)==1) = 0;
end