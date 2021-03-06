<center><img src="enigma.png"></center>
<center>New Orleans Abatement Data Cleaning</center>
---------------

## Navigation
  *  [Overview](../index.html)
  *  [Docs](../docs.html)
  *  [Analysis](analysis.html)
  *  [_Download this repository_](nola.zip)


### Setup

```r
rm(list=ls())
options(scipen=999)
require(gdata)
```

```
## Loading required package: gdata
## gdata: read.xls support for 'XLS' (Excel 97-2004) files ENABLED.
## 
## gdata: read.xls support for 'XLSX' (Excel 2007+) files ENABLED.
## 
## Attaching package: 'gdata'
## 
## The following object is masked from 'package:stats':
## 
##     nobs
## 
## The following object is masked from 'package:utils':
## 
##     object.size
```

```r
require(stringr)
```

```
## Loading required package: stringr
```

```r
require(plyr)
```

```
## Loading required package: plyr
```

```r
require(lubridate)
```

```
## Loading required package: lubridate
## Loading required package: methods
## 
## Attaching package: 'lubridate'
## 
## The following object is masked from 'package:plyr':
## 
##     here
```

### Parameters:

```r
wd <- '/Users/brian/enigma/analytics-projects/nola/' # set your working directory here.
setwd(wd)
infile <- paste0(wd, 'data/abatement_review_data_raw.xls')
outfile <- paste0(wd, 'data/clean_abatement_data.Rdata')
```

### Read in excel files

```r
d <- read.xls(
  infile,
  sheet = 1,
  header = T, 
  na.strings=c("N/A", 'NA'), 
  blank.lines.skip=T,
  perl='/usr/bin/perl'
)

rev <- read.xls(
  paste0(wd, 'data/all_cases_reviewers.xls'),
  sheet = 1,
  header = T,
  na.strings=c("N/A", 'NA'),
  blank.lines.skip=T,
  perl='/usr/bin/perl'
)

geo <- read.csv(
  paste0(wd, 'data/nola_abatement_geo.csv'), 
  as.is=T
)

census <- read.csv(
  paste0(wd, 'data/census_data.csv'), 
  as.is=T
)
```

### Clean up column names

```r
slugify <- function(col_names) {
  col_names <- tolower( gsub( '\\.', '_', str_trim(col_names) ) )
  return( gsub('(_)$|^(_)', '', col_names) )
}
names(d) <- slugify(names(d))
names(rev) <- slugify(names(rev))

#junk from excel 
d$x <- NULL
d$x_1 <- NULL
d$x_2 <- NULL
```

### Cleanup fields

```r
# helper functions
std_bool <- function(x){
  x <- tolower( as.character(x) )
  x[x=="yes"] <- 1 
  x[x=="no"] <- 0
  x[x=="0"] <- 0 
  x[x=="1"] <- 1
  return( as.numeric(x) )
}

std_code <- function(x) {
  only_digits <- gsub( '[^0-9]+', '', as.character(x) )
  return( as.numeric(only_digits) )
}

std_desc <- function(x) {
  return (str_trim(gsub( '[^A-Za-z ]+', '', as.character(x) )))
}

# hearing date
d$hearing_date <- mdy(d$hearing_date)
```

```
## Warning: 7 failed to parse.
```

```r
# mva clusters
d$mva_cluster <- as.character(d$mva_cluster)
valid_mva_clusters <- c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H')
d <- d[(d$mva_cluster %in% valid_mva_clusters),]

# mva codes
d$mva_code <- as.character(d$mva_code)
valid_mva_codes <- c('1', '2', '3', '4', '5', '6', '7', '8')
d <- d[(d$mva_code %in% valid_mva_codes),]
d$mva_code <- as.numeric(d$mva_code)

# std. bool - vacant lot 
d$vacant_lot <- std_bool(d$vacant_lot)

# std. bool - vacant lot revised 
d$vacant_lot_revised_code <- std_bool(d$vacant_lot_revised)
d$vacant_lot_revised <- NULL

# fill in NA's of vacant lot code 
d$vacant_lot_code <- std_bool(d$vacant_lot)

# market assessments
d$market_assessment <- std_desc(d$market_assessment)
for (i in c('Very Strong', 'Moderately Strong', 'Moderately Weak', 'Very Weak')) {
  d$market_assessment[grep(i, d$market_assessment)] <- i
}
d$market_assessment <- as.factor(d$market_assessment)

# market code 
d$market_code <- as.numeric(d$market_code)

# cleanup roof codes 
d$roof <- std_desc(d$roof)
d$roof_code <- std_code(d$roof_code)

# cleanup exterior codes
d$exterior <- std_desc(d$exterior)
d$exterior_code <- std_code(d$exterior_code)

# cleanup foundation codes 
d$foundation <- std_desc(d$foundation)
d$foundation_code <- std_code(d$foundation_code)

# cleanup overall condition code  
d$overall_condition <- std_desc(d$overall_condition)
d$overall_condition_code <- std_code(d$overall_condition_code)

# cleanup char code 
d$character <- std_bool(d$character)
d$character_code <- as.numeric(d$character)

# cleanup err codes 
d$potential_error_in_lot_status <- as.factor(as.numeric(d$potential_error_in_lot_status))
d$potential_error_in_market_status <- as.factor(as.numeric(d$potential_error_in_market_status))

# Cleanup rec code
d$abatement_recommendation <- tolower(as.character(d$abatement_recommendation))
d$recommendation_code <- as.numeric(d$recommendation_code)
d$recommendation_code[d$abatement_recommendation=='sell'] <- 1 
d$recommendation_code[d$abatement_recommendation=='demolish'] <- 0
d$abatement_recommendation <- as.factor(d$abatement_recommendation)
```

### Join reviewer codes

```r
d$case_number <- as.character(d$case_number)
rev$case_number <- as.character(rev$case_number)
rev <- subset(rev, select=c('case_number', 'reviewer'))
d <- join(rev, d, type='right', by='case_number')
```

### Join geo data

```r
d$case_number <- as.character(d$case_number)
geo$case_number <- as.character(geo$case_num)
geo$councildis <- as.factor(geo$councildis)
geo$case_num <- NULL
geo <- subset(geo, select=c('case_number', 'geoid', 'longitude', 'latitude', 'councildis', 'precinctid'))
d <- join(geo, d, type="right", by="case_number")
```

### Join census data

```r
census$geoid <- census$census_geoid
census$census_geoid <- NULL
census$median_rent <- census$census_median_contract_001_median_contract_rent
census$population <- census$census_population_001_total
census$census_median_contract_001_median_contract_rent <- NULL
census$census_population_001_total <- NULL
d <- join(census, d, type="right", by="geoid")
d <- d[!duplicated(d$case_number), ]
```

### Store clean data

```r
save(d, file=outfile)
write.csv(d, gsub('Rdata', 'csv', outfile), row.names=F)
```

