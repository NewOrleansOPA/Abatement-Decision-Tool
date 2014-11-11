<center><img src="reports/enigma.png"></center>
<center>New Orleans Abatement Analytics</center>
==============


Brian Abelson, Enigma, 2014


## Navigation

  *  [Overview](index.html)
  *  [Data Cleaning](reports/munge.html)
  *  [Analysis](reports/analysis.html)
  *  [_Download this repository_](nola.zip)


All management of the project is done through the [`Makefile`](Makefile)

## Install Required Dependencies
Runs `rscript/requirements.r` and `pip install -r requirements.txt`

```bash
$ make init
```

## (Optionally) Download Census Data
via `pyscripts/get_census_data`
**NOTE** The output of this script is already located at [`data/census-data.csv`](`data/census-data.csv`).

```bash
$ make census_data
```

## Clean and join data
via `rscripts/munge.Rmd`
**NOTE** This is an [RMarkdown](http://rmarkdown.rstudio.com/) document.
```bash
$ make munge
```

## Run the analysis pipline
via `rscripts/analysis.Rmd`
**NOTE** This is an [RMarkdown](http://rmarkdown.rstudio.com/) document.
```bash
$ make analysis
```

## Render the documentation
Converts [`Index.md`](index.md) => [`index.html`](index.html)
Converts [`README.md`](README.md) => [`docs.html`](docs.html)
```bash
$ make docs
```

## View the outputted documentation
**NOTE** This is an [RMarkdown](http://rmarkdown.rstudio.com/) document.
```bash
$ make view
```

## Utilities

* `make build`
  - Syncs the output on an s3 website
* `make compress`
  - Creates a compressed archive of this repository
* `make all`
  - Runs the entire analysis pipeline.
* `make push`
  - Pushes the repository to master + gh-pages branches.

## TODO

- [x] Cleanup Data
- [x] Deal with missing values via Imputation
- [x] Random Forest
= [x] Logistic Regression
- [x] Decision Tree 
- [x] Naive Bayes 
- [x] Ensemble 
- [x] Model Selection
- [x] Model Analysis 
- [x] Analyze coder disparities 
- [x] Put It On A Map
  * [http://enigmaio.cartodb.com/viz/a4364e76-5a34-11e4-a921-0e853d047bba/public_map](http://enigmaio.cartodb.com/viz/a4364e76-5a34-11e4-a921-0e853d047bba/public_map)

## SQL QUERIES
### AGG JOINS
```sql
SELECT 
  case_number_agg.p_demolish, 
  case_number_agg.p_sell, 
  case_number_agg.error, 
  case_number_agg.variance, 
  nola_abatement_geo.the_geom,
  nola_abatement_geo.case_number 
FROM case_number_agg, case_number 
WHERE nola_abatement_geo.case_number = case_number_agg.case_number 
```

## Original Briefing

The data is a set of 603 cases where a review team of mid-level code enforcement supervisors assigned values according to 10 different variables which were identified during interviews with code enforcement staff and leadership to have salience in the decision of whether a property with a blight judgment should proceed down the path of code lien foreclosure (or “sell”) or demolition. That team then also made a recommendation on which abatement path that property should pursue (sell or demolish).

What we would like to do is use this data to create a beta decision support tool to make a recommendation to code enforcement leadership on an abatement decision to make those decisions more consistent and swift. Ideally, we would refresh the model on a periodic basis once we have more data on whether or not code leadership rejected or accepted the recommendation. Over time, we hope the model will closely replicate the decision making process of the SMEs.

One way to approach this analysis is to create an abatement decision “scorecard” where each of the ten independent variables are assigned some weight, and if the totals of a score exceeds some threshold, the tool will make a recommendation to demolish; otherwise, the recommendation is to sell. Mainly because this approach is within our capacity, we can develop a model using this approach

But I am sure there are other approaches, such as developing a decision tree where certain variables may be conditional upon the assignment of other variables. There may be many other approaches as well. Where we could use your help is to develop an alternative model to the one using logistic regression so we can then test our options to see which works best.

There are some data integrity issues, which are noted in the “notes” tab of the spreadsheet. There are also some questions about how often the recommendation of the review team is actually consistent with the determination code enforcement leadership would have taken. However, for a beta model, I think this data is probably good enough to start and as we get more data, we can improve it over time.

We are R users over here (mainly because we’re cheap). Best case scenario is you all use R as well and then we can work from each other’s code. We’ll put our code for the linear regression on Github so you can see what we’re up to.  If you all don’t use R, no big deal – just tell us what you did and we can just rewrite the code later on.

If you or your analysts have any questions, please let me know.

Thanks again. We very much appreciate your help.

