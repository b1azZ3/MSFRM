library(ggplot2)
library(ggthemes)
library(lubridate)
library(dplyr)
library(tidyr)
library(tidyverse) # metapackage of all tidyverse packages
library(DT)
library(scales)

colors = c("#CC1011", "#665555", "#05a399", "#cfcaca", "#f5e840", "#0683c9", "#e075b0")
colors

# Read the data for each month separately
apr <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-apr14.csv')
may <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-may14.csv')
june <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-jun14.csv')
july <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-jul14.csv')
aug <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-aug14.csv')
sep <- read.csv('D:/VF/R-projects/uber-raw_data/uber-raw-data-sep14.csv')

# Combine the data together
data <- rbind(apr, may, june, july, aug, sep)
cat('The dimensions of data are:', dim(data))

# Print the first 6 rows of data
head(data)

data$Date.Time <- as.POSIXct(data$Date.Time, format="%m/%d/%Y %H:%M:%S")
data$Time <- format(as.POSIXct(data$Date.Time, format = "%m/%d/%Y %H:%M:%S"), format="%H:%M:%S")
data$Date.Time <- ymd_hms(data$Date.Time)

# Create individual columns for month and year and time variables
data$day <- factor(day(data$Date.Time))
data$month <- factor(month(data$Date.Time, label=TRUE))
data$year <- factor(year(data$Date.Time))
data$dayofweek <- factor(wday(data$Date.Time, label=TRUE))
data$second <- factor(second(hms(data$Time)))
data$minute <- factor(minute(hms(data$Time)))
data$hour <- factor(hour(hms(data$Time)))
head(data)

## Data Visualization
# Plot the trips by hours in a day
hourly_data <- data %>%
                    group_by(hour) %>%
                                   dplyr::summarize(Total = n())
datatable(hourly_data)

# Plot the data by hour
ggplot(hourly_data, aes(hour, Total)) +
  geom_bar(stat='identity',
           fill='steelblue',
           color='red') + 
  ggtitle("Trips Every hour", subtitle = "aggregate today") + 
  theme(legend.position = 'none',
        plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5)) + 
  scale_y_continuous(labels=comma)

# Plot trips by hour and month
month_hour_data <- data %>%
  group_by(month, hour) %>%
  dplyr::summarise(Total = n())

ggplot(month_hour_data, aes(hour, Total, fill = month)) + 
  geom_bar(stat = "identity") + 
  ggtitle("Trips By Hour and Month") + 
  scale_y_continuous(labels = comma)

# Plot the data by trips during every data of the month
day_data <- data %>% 
  group_by(day) %>%
  dplyr::summarise(Trips = n())

ggplot(day_data, aes(day, Trips)) +
  geom_bar(stat = 'identity', fill = 'steelblue') +
  ggtitle("Trips by day of the month") +
  theme(legend.position = 'none') +
  scale_y_continuous(labels = comma)

# Plot data by day of the week and month
day_month_data <- data %>%
  group_by(dayofweek, month) %>%
  dplyr::summarise(Trips = n())

ggplot(day_month_data, aes(dayofweek, Trips, fill = month)) +
  geom_bar(stat='identity', aes(fill = month), position = "dodge") +
  ggtitle("Trips by Day and Month") +
  scale_y_continuous(labels = comma) +
  scale_fill_manual(values = colors)

# Plot data by Trips during month in a year
month_data <- data %>%
  group_by(month) %>%
  dplyr::summarise(Total = n())

ggplot(month_data, aes(month, Total, fill = month)) +
  geom_bar(stat = "Identity") + 
  ggtitle("Trips in a month") + 
  theme(legend.position = "none") + 
  scale_y_continuous(labels = comma) + 
  scale_fill_manual(values = colors)

# Heatmap by Hour and Day
day_hour_data <- data %>%
  group_by(day, hour) %>%
  dplyr::summarise(Total = n())
datatable(day_hour_data)

ggplot(day_hour_data, aes(day, hour, fill = Total)) +
  geom_tile(color = "white") +
  ggtitle("Heatmap by Hour and Day")

# Heatmap by dayofweek and month
ggplot(day_month_data, aes(dayofweek, month, fill = Trips)) + 
  geom_tile(color = "white") + 
  ggtitle("Heat Map by Month and Day")

# Creating a map visualization of rides in NYC
# Set map constant
min_lat <- 40 
max_lat <- 40.91
min_long <- -74.15
max_long <- -73.7004

ggplot(data, aes(x=Lon, y=Lat)) +
  geom_point(size=1, color="blue") +
  scale_x_continuous(limits=c(min_long, max_long)) +
  scale_y_continuous(limits=c(min_lat, max_lat)) +
  theme_map() +
  ggtitle("NYC MAP BASED ON UBER RIDES DURING 2014 (APR-SEP)")

