## Script to read in Danny, Greg, and Johns classification into single table for exploration

require(tidyverse)
require(openxlsx)

d1 <- read.xlsx('datavalidation/dataForDanny.xlsx') %>% mutate(`yes/no` = tolower(`yes/no`),
                                                               sport = tolower(sport))
d2 <- read.xlsx('datavalidation/dataForGregcomplete.xlsx') %>% mutate(`yes/no` = tolower(`yes/no`),
                                                                      sport = tolower(sport))
d3 <- read.xlsx('datavalidation/dataForJohn_2.xlsx') %>% mutate(`yes/no` = tolower(`yes/no`),
                                                                sport = tolower(sport))

#str(d1)
#str(d2)
#str(d3)
d<- bind_rows(d1,d2,d3) %>%
  arrange(url)


table(d$`yes/no`)

d %>%
  filter(`yes/no` == 'yes',
         sport == 'soccer')


## keep dup urls and compare classifications
d_dups <- 
  d %>%
  group_by(rec_number) %>%
  filter(n() > 1) %>%
  ungroup() %>%
  select(`yes/no`, sport, rec_number, url, title, owner)

## mark non identical records
##
## Given a specific record ID that is duplicated
## When the `yes\no` and sport fields are not the same for each entry
## Then mark that record with unmatched = 1

df_iden <- NULL
for(rID in unique(d_dups$rec_number)) {
  thisID = d_dups %>% filter(rec_number == rID)
  if(length(unique(thisID$`yes/no`)) == 1 &
     length(unique(thisID$sport)) == 1) {
    identical = 1 } else {identical = 0
  }
  df_iden <- bind_rows(df_iden, data_frame(rec_number = rID, identical))
}

#

## duplicated initial ids
left_join(d_dups, df_iden) %>%
  write.xlsx(file = 'datavalidation/dupCompare.xlsx')


## data for test train - eliminate summary, position, owner and any dups
#library(urltools)
d_model <- d %>%
  select(-summary, -position, -owner, -rec_number) %>%
  distinct() %>%
  rename(goodPage = `yes/no`) 

## Domain process
## 1) if the domain has a www then strip
## 2) remove the last id (com/net/org)
for(r in seq(1:nrow(d_model))) {
  uri = unlist(strsplit(d_model$abrev_domain[r] , '[.]'))
  adj_uri = uri[1:(length(uri)-1)]
  if(adj_uri[1] == 'www') {
    # get rid of it
    adj_uri = adj_uri[2:length(adj_uri)]
  }
  d_model$abrev_domain2[r] = str_flatten(adj_uri, ".")
  # if(grepl('www', d_model$abrev_domain[r])) {
  #   
  #   d_model$abrev_domain[r] = paste(uri[2],uri[3],sep=".")
  # }
}


write.csv(d_model, file = 'datavalidation/dataToTrainModel.csv', row.names = F)
