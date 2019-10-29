library(dplyr)
data = read.csv('data_for_classification_ori.csv')
df = data %>%
  transmute(promotion = promotion_max
            ,promotion_cnt
            ,year = year_last
            ,month = month_num_last
            ,months_since_promotion_max
            ,comparatio_last
            #,span_of_control_direct_last
            #,span_of_control_total_last
            #,movement_lateral_cnt_sum_last
            #,movement_demote_event_cnt_sum_last
            #,transfer_event_cnt_sum_last
            #,changed_spv_cnt_sum_last
            ,awards_points_cnt_sum_last
            #,awards_points_sum_last
            ,awards_bonus_cnt_sum_last
            #,awards_bonus_sum_last
            ,awards_peer_cnt_sum_last
            #,reporting_level_last
            ,supervisor_Y_last
            ,tenure_in_months_last
            ,region_America = site_country_region_last == 'Americas'
            ,region_Asia = site_country_region_last == 'Asia'
  ) 
write.csv(df,file='data_for_classification.csv',row.names=FALSE)
