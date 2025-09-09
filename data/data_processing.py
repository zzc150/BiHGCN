#
import pandas as pd

# # 读取文件
# taxi_file_path = './mydata/nyctaxi/taxiflowin4-7.csv'
# bike_file_path = './mydata/nycbike/3-7bikeflowin.csv'
# # 加载数据
# taxi_data = pd.read_csv(taxi_file_path)
# bike_data = pd.read_csv(bike_file_path)
#
# # 转换时间格式
# bike_data['time_slot'] = pd.to_datetime(bike_data['time_slot'])
# taxi_data['time_slot'] = pd.to_datetime(taxi_data['time_slot'])
#
# # 提取bike数据中的所有LocationID
# bike_location_ids = bike_data['LocationID'].unique()
# # bike_location_ids = pd.read_csv('./nycbike/bike_location_ids.csv')
# # 删除7月1日的数据
# taxi_data = taxi_data[taxi_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
#
# # 使用DOLocationID列进行筛选
# filtered_taxi_data = taxi_data[taxi_data['DOLocationID'].isin(bike_location_ids)]
#
# # 删除时间列并转换为N×T的形状
# bike_pivot = bike_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
# taxi_pivot = filtered_taxi_data.pivot(index='DOLocationID', columns='time_slot', values='taxi_count')
#
# # 填充缺失值并转换为浮点数
# bike_pivot = bike_pivot.fillna(0).astype(float)
# taxi_pivot = taxi_pivot.fillna(0).astype(float)
#
# # # 保存区域ID
# # bike_location_ids_file_path = './nycbike/bikeout_location_ids.csv'
# # taxi_location_ids_file_path = './nyctaxi/taxiout_location_ids.csv'
# # pd.DataFrame(bike_pivot.index).to_csv(bike_location_ids_file_path, header=False, index=False, float_format='%.6f')
# # pd.DataFrame(taxi_pivot.index).to_csv(taxi_location_ids_file_path, header=False, index=False, float_format='%.6f')
#
# # 保存转换后的数据
# # bike_pivot_file_path = './mydata/nycbike/bike_flowout.csv'
# taxi_pivot_file_path = './mydata/nyctaxi/taxi_flowin.csv'
# # bike_pivot.to_csv(bike_pivot_file_path, header=False, index=False, float_format='%.6f')
# taxi_pivot.to_csv(taxi_pivot_file_path, header=False, index=False, float_format='%.6f')
# # print(f"Bike flow data saved to: {bike_pivot_file_path}")
# print(f"Taxi flow data saved to: {taxi_pivot_file_path}")
# print(f"Bike location IDs saved to: {bike_location_ids_file_path}")
# print(f"Taxi location IDs saved to: {taxi_location_ids_file_path}")

#
#
# taxi_pick = pd.read_csv(r'.\nyctaxi\taxi_pick.csv', index_col=None, header=None).values  # L, N
# print(taxi_pick.shape)
#
# taxi_drop = pd.read_csv(r'.\nyctaxi\taxi_drop.csv', index_col=None, header=None).values
# print(taxi_drop.shape)
# bike_pick = pd.read_csv(r'.\nycbike\bike_pick.csv', index_col=None, header=None).values
# print(bike_pick.shape)
# bike_drop = pd.read_csv(r'.\nycbike\bike_drop.csv', index_col=None, header=None).values
# print(bike_drop.shape)
#
# df1 = pd.read_csv(r'.\mydata\nycbike\bike_flowin.csv', index_col=None, header=None).values
# print(df1.shape)
# df2 = pd.read_csv(r'.\mydata\nycbike\bike_flowout.csv', index_col=None, header=None).values
# print(df2.shape)
# df3 = pd.read_csv(r'.\mydata\nyctaxi\taxi_flowin.csv', index_col=None, header=None).values
# print(df3.shape)
# #
# df4 = pd.read_csv(r'.\mydata\nyctaxi\taxi_flowout.csv', index_col=None, header=None).values
# print(df4.shape)


#把taxi按照bike的区域进行筛选，taxipick和taxidrop，bikepick和bikedrop的区域相同，有缺失的bike
# import pandas as pd
#
# # 读取文件
# taxi_dropoff_file_path = './mydata/nyctaxi/taxiflowin4-7.csv'
# taxi_pickup_file_path = './mydata/nyctaxi/taxiflowout4-7.csv'
# bike_dropoff_file_path = './mydata/nycbike/3-7bikeflowin.csv'
# bike_pickup_file_path = './mydata/nycbike/3-7bikeflowout.csv'
#
# # # 加载数据
# taxi_dropoff_data = pd.read_csv(taxi_dropoff_file_path)
# taxi_pickup_data = pd.read_csv(taxi_pickup_file_path)
# bike_dropoff_data = pd.read_csv(bike_dropoff_file_path)
# bike_pickup_data = pd.read_csv(bike_pickup_file_path)
#
# # 转换时间格式
# bike_dropoff_data['time_slot'] = pd.to_datetime(bike_dropoff_data['time_slot'])
# taxi_dropoff_data['time_slot'] = pd.to_datetime(taxi_dropoff_data['time_slot'])
# bike_pickup_data['time_slot'] = pd.to_datetime(bike_pickup_data['time_slot'])
# taxi_pickup_data['time_slot'] = pd.to_datetime(taxi_pickup_data['time_slot'])
#
# # 提取bike dropoff数据中的所有LocationID
# bike_dropoff_location_ids = bike_dropoff_data['LocationID'].unique()
# bike_pickup_location_ids = bike_pickup_data['LocationID'].unique()
# taxi_pickup_location_ids = taxi_pickup_data['PULocationID'].unique()
# taxi_dropoff_location_ids = taxi_dropoff_data['DOLocationID'].unique()
#
# # 根据bike dropoff的LocationID筛选bike pickup数据
# bike_pickup_data_filtered = bike_pickup_data[bike_pickup_data['LocationID'].isin(bike_dropoff_location_ids)]
#
# # 对齐bike pickup数据中的LocationID
# bike_pickup_data_aligned = bike_dropoff_data[['LocationID', 'time_slot']].merge(
#     bike_pickup_data_filtered,
#     on=['LocationID', 'time_slot'],
#     how='left'
# ).fillna(0)
#
# bike_pickup_pivot = bike_pickup_data_aligned.pivot(index='LocationID', columns='time_slot', values='ride_count')
# bike_pickup_pivot = bike_pickup_pivot.fillna(0).astype(float)
# bike_pickup_pivot_file_path = 'mydata/nycbike/bike_flowout.csv'
# bike_pickup_pivot.to_csv(bike_pickup_pivot_file_path, header=False, index=False)
# print(f"Bike pickup flow data saved to: {bike_pickup_pivot_file_path}")
# # 删除7月1日的数据
# taxi_dropoff_data = taxi_dropoff_data[taxi_dropoff_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
# taxi_pickup_data = taxi_pickup_data[taxi_pickup_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
#
# 使用DOLocationID列进行筛选
# filtered_taxi_dropoff_data = taxi_dropoff_data[taxi_dropoff_data['DOLocationID'].isin(bike_dropoff_location_ids)]
# filtered_taxi_pickup_data = taxi_pickup_data[taxi_pickup_data['PULocationID'].isin(bike_dropoff_location_ids)]
# 删除时间列并转换为N×T的形状
# bike_dropoff_pivot = bike_dropoff_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
# # taxi_dropoff_pivot = filtered_taxi_dropoff_data.pivot(index='DOLocationID', columns='time_slot', values='taxi_count')
#
# bike_pickup_pivot = bike_pickup_data(index='LocationID', columns='time_slot', values='ride_count')
# # taxi_pickup_pivot = filtered_taxi_pickup_data.pivot(index='PULocationID', columns='time_slot', values='taxi_count')
#
# # 填充缺失值（如果有的话）
# bike_dropoff_pivot = bike_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
# # taxi_dropoff_pivot = taxi_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
#
# bike_pickup_pivot = bike_pickup_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
# # taxi_pickup_pivot = taxi_pickup_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
#
# # # 保存区域ID
# # bike_location_ids_file_path = './nycbike/bike_location_ids.csv'
# # taxi_location_ids_file_path = './nyctaxi/taxi_location_ids.csv'
# # pd.DataFrame(bike_dropoff_pivot.index).to_csv(bike_location_ids_file_path, header=False, index=False)
# # pd.DataFrame(taxi_dropoff_pivot.index).to_csv(taxi_location_ids_file_path, header=False, index=False)
#
# # 保存转换后的数据，确保数据为浮点数格式
# # bike_dropoff_pivot_file_path = './nycbike/bike_dropoff_flowin.csv'
# # taxi_dropoff_pivot_file_path = './nyctaxi/taxi_dropoff_flowin.csv'
# bike_pickup_pivot_file_path = './nycbike/bike_flowout.csv'
# # taxi_pickup_pivot_file_path = './nyctaxi/taxi_pickup_flowin.csv'
#
# # bike_dropoff_pivot.to_csv(bike_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
# # taxi_dropoff_pivot.to_csv(taxi_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
# bike_pickup_pivot.to_csv(bike_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')
# # taxi_pickup_pivot.to_csv(taxi_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')
#
# # print(f"Bike dropoff flow data saved to: {bike_dropoff_pivot_file_path}")
# # print(f"Taxi dropoff flow data saved to: {taxi_dropoff_pivot_file_path}")
# print(f"Bike pickup flow data saved to: {bike_pickup_pivot_file_path}")
# # print(f"Taxi pickup flow data saved to: {taxi_pickup_pivot_file_path}")
# # print(f"Bike location IDs saved to: {bike_location_ids_file_path}")
# # print(f"Taxi location IDs saved to: {taxi_location_ids_file_path}")







#taxi和bike具有不同的区域数
# import pandas as pd
#
# # 读取文件
# taxi_dropoff_file_path = './mydata/nyctaxi/taxiflowin4-7.csv'
# taxi_pickup_file_path = './mydata/nyctaxi/taxiflowout4-7.csv'
# bike_dropoff_file_path = './mydata/nycbike/3-7bikeflowin.csv'
# bike_pickup_file_path = './mydata/nycbike/3-7bikeflowout.csv'
#
# # # 加载数据
# taxi_dropoff_data = pd.read_csv(taxi_dropoff_file_path)
# taxi_pickup_data = pd.read_csv(taxi_pickup_file_path)
# bike_dropoff_data = pd.read_csv(bike_dropoff_file_path)
# bike_pickup_data = pd.read_csv(bike_pickup_file_path)
#
# # 转换时间格式
# bike_dropoff_data['time_slot'] = pd.to_datetime(bike_dropoff_data['time_slot'])
# taxi_dropoff_data['time_slot'] = pd.to_datetime(taxi_dropoff_data['time_slot'])
# bike_pickup_data['time_slot'] = pd.to_datetime(bike_pickup_data['time_slot'])
# taxi_pickup_data['time_slot'] = pd.to_datetime(taxi_pickup_data['time_slot'])
#
# # 提取bike dropoff数据中的所有LocationID
# bike_dropoff_location_ids = bike_dropoff_data['LocationID'].unique()
# bike_pickup_location_ids = bike_pickup_data['LocationID'].unique()
# taxi_pickup_location_ids = taxi_pickup_data['PULocationID'].unique()
# taxi_dropoff_location_ids = taxi_dropoff_data['DOLocationID'].unique()
#
# # # 删除7月1日的数据
# # taxi_dropoff_data = taxi_dropoff_data[taxi_dropoff_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
# # taxi_pickup_data = taxi_pickup_data[taxi_pickup_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
# # bike_dropoff_data = bike_dropoff_data[bike_dropoff_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
# # bike_pickup_data = bike_pickup_data[bike_pickup_data['time_slot'].dt.date < pd.to_datetime('2023-07-01').date()]
#
# # 使用LocationID列进行筛选
# filtered_bike_dropoff_data = bike_dropoff_data[bike_dropoff_data['LocationID'].isin(bike_pickup_location_ids)]
# # filtered_bike_pickup_data = bike_pickup_data[bike_pickup_data['LocationID'].isin(bike_pickup_location_ids)]
#
# # 删除时间列并转换为N×T的形状
# bike_dropoff_pivot = filtered_bike_dropoff_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
# bike_pickup_pivot = bike_pickup_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
# taxi_dropoff_pivot = taxi_dropoff_data.pivot(index='DOLocationID', columns='time_slot', values='taxi_count')
# taxi_pickup_pivot = taxi_pickup_data .pivot(index='PULocationID', columns='time_slot', values='taxi_count')
#
# # 填充缺失值（如果有的话）
# # bike_dropoff_pivot = bike_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
# # taxi_dropoff_pivot = taxi_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
#
#
#
#
# # 保存区域ID
# bikein_location_ids_file_path = './mydata/nycbike/bikein_location_ids.csv'
# bikeout_location_ids_file_path = './mydata/nycbike/bikeout_location_ids.csv'
# taxi_location_ids_file_path = './mydata/nyctaxi/taxi_location_ids.csv'
# pd.DataFrame(bike_dropoff_pivot.index).to_csv(bikein_location_ids_file_path, header=False, index=False)
# pd.DataFrame(bike_pickup_pivot.index).to_csv(bikeout_location_ids_file_path, header=False, index=False)
# pd.DataFrame(taxi_dropoff_pivot.index).to_csv(taxi_location_ids_file_path, header=False, index=False)
#
# # 保存转换后的数据，确保数据为浮点数格式
# bike_dropoff_pivot_file_path = './mydata/nycbike/bike_drop.csv'
# taxi_dropoff_pivot_file_path = './mydata/nyctaxi/taxi_drop.csv'
# bike_pickup_pivot_file_path = './mydata/nycbike/bike_pick.csv'
# taxi_pickup_pivot_file_path = './mydata/nyctaxi/taxi_pick.csv'
#
# bike_dropoff_pivot.to_csv(bike_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
# taxi_dropoff_pivot.to_csv(taxi_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
# bike_pickup_pivot.to_csv(bike_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')
# taxi_pickup_pivot.to_csv(taxi_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')



#曼哈顿的流量数据
import pandas as pd
import geopandas as gpd
# 读取文件
taxi_dropoff_file_path = './mydata/nyctaxi/taxiflowin4-7.csv'
taxi_pickup_file_path = './mydata/nyctaxi/taxiflowout4-7.csv'
bike_dropoff_file_path = './mydata/nycbike/3-7bikeflowin.csv'
bike_pickup_file_path = './mydata/nycbike/3-7bikeflowout.csv'
taxi_map = gpd.read_file(r'F:\zzc\模型源代码\GSABT-master\GSABT-master\data\taxi_zones.zip')
taxi_map.at[56, 'LocationID'] = 57
taxi_map.at[102, 'LocationID'] = 105
taxi_map.at[103, 'LocationID'] = 105
taxi_map.at[104, 'LocationID'] = 105
print(taxi_map.info)
# 筛选 borough 列为 Manhattan 的行，并提取对应的 LocationID
manhattan_locations = taxi_map[taxi_map['borough'] == 'Manhattan']['LocationID']
# 删除特定的LocationID
ids_to_remove = {105, 153}
manhattan_locations = manhattan_locations[~manhattan_locations.isin(ids_to_remove)]
# # 加载数据
taxi_dropoff_data = pd.read_csv(taxi_dropoff_file_path)
taxi_pickup_data = pd.read_csv(taxi_pickup_file_path)
bike_dropoff_data = pd.read_csv(bike_dropoff_file_path)
bike_pickup_data = pd.read_csv(bike_pickup_file_path)

# 转换时间格式
bike_dropoff_data['time_slot'] = pd.to_datetime(bike_dropoff_data['time_slot'])
taxi_dropoff_data['time_slot'] = pd.to_datetime(taxi_dropoff_data['time_slot'])
bike_pickup_data['time_slot'] = pd.to_datetime(bike_pickup_data['time_slot'])
taxi_pickup_data['time_slot'] = pd.to_datetime(taxi_pickup_data['time_slot'])



# 使用LocationID列进行筛选
filtered_bike_dropoff_data = bike_dropoff_data[bike_dropoff_data['LocationID'].isin(manhattan_locations)]
filtered_bike_pickup_data = bike_pickup_data[bike_pickup_data['LocationID'].isin(manhattan_locations)]
filtered_taxi_pickup_data = taxi_pickup_data[taxi_pickup_data['PULocationID'].isin(manhattan_locations)]
filtered_taxi_dropoff_data = taxi_dropoff_data[taxi_dropoff_data['DOLocationID'].isin(manhattan_locations)]
# 提取bike dropoff数据中的所有LocationID
bike_dropoff_location_ids = filtered_bike_dropoff_data['LocationID'].unique()
bike_pickup_location_ids = filtered_bike_pickup_data['LocationID'].unique()
taxi_pickup_location_ids = filtered_taxi_pickup_data['PULocationID'].unique()
taxi_dropoff_location_ids = filtered_taxi_dropoff_data['DOLocationID'].unique()


# 删除时间列并转换为N×T的形状
bike_dropoff_pivot = filtered_bike_dropoff_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
bike_pickup_pivot = filtered_bike_pickup_data.pivot(index='LocationID', columns='time_slot', values='ride_count')
taxi_dropoff_pivot = filtered_taxi_dropoff_data.pivot(index='DOLocationID', columns='time_slot', values='taxi_count')
taxi_pickup_pivot = filtered_taxi_pickup_data.pivot(index='PULocationID', columns='time_slot', values='taxi_count')

# 填充缺失值（如果有的话）
# bike_dropoff_pivot = bike_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)
# taxi_dropoff_pivot = taxi_dropoff_pivot.reindex(bike_dropoff_location_ids).fillna(0).astype(float)




# 保存区域ID
bikein_location_ids_file_path = './mydata/manhadun/bike/bike_location_ids.csv'
# bikeout_location_ids_file_path = './mydata/manhadun/bike/bikeout_location_ids.csv'
taxi_location_ids_file_path = './mydata/manhadun/taxi/taxi_location_ids.csv'
manhattan_locations_file_path = './mydata/manhadun/manhattan_locations.csv'
pd.DataFrame(bike_dropoff_pivot.index).to_csv(bikein_location_ids_file_path, header=False, index=False)
# pd.DataFrame(bike_pickup_pivot.index).to_csv(bikeout_location_ids_file_path, header=False, index=False)
pd.DataFrame(taxi_dropoff_pivot.index).to_csv(taxi_location_ids_file_path, header=False, index=False)
pd.DataFrame(manhattan_locations).to_csv(manhattan_locations_file_path, header=False, index=False)
# 保存转换后的数据，确保数据为浮点数格式
bike_dropoff_pivot_file_path = './mydata/manhadun/bike/bike_drop.csv'
taxi_dropoff_pivot_file_path = './mydata/manhadun/taxi/taxi_drop.csv'
bike_pickup_pivot_file_path = './mydata/manhadun/bike/bike_pick.csv'
taxi_pickup_pivot_file_path = './mydata/manhadun/taxi/taxi_pick.csv'

bike_dropoff_pivot.to_csv(bike_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
taxi_dropoff_pivot.to_csv(taxi_dropoff_pivot_file_path, header=False, index=False, float_format='%.6f')
bike_pickup_pivot.to_csv(bike_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')
taxi_pickup_pivot.to_csv(taxi_pickup_pivot_file_path, header=False, index=False, float_format='%.6f')

