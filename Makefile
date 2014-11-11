init:
	Rscript rscripts/requirements.r
	sudo pip install -r pyscripts/requirements.txt

# build:
# 	s3cmd sync ../nola s3://enigma-analytics/
# 	s3cmd setacl -r s3://enigma-analytics/nola/ --acl-public

compress:

	zip -r nola.zip ../nola

census_data:

	python pyscripts/get_census_data.py

munge:

	# Munge the raw data
	cd reports/ && \
	Rscript -e 'knitr::knit("../rscripts/munge.Rmd", "../reports/munge.md")'

	Rscript -e 'markdown::markdownToHTML("reports/munge.md", "reports/munge.html")'

analysis:

	cd reports/ && \
	Rscript -e 'knitr::knit("../rscripts/analysis.Rmd", "../reports/analysis.md")'
	Rscript -e 'markdown::markdownToHTML("reports/analysis.md", "reports/analysis.html")'

docs:

	Rscript -e 'markdown::markdownToHTML("Index.md", "index.html")'	
	Rscript -e 'markdown::markdownToHTML("README.md", "docs.html")'	

view:

	python -m SimpleHTTPServer

all:
	make init
	make census_data
	make munge
	make analysis
	make docs
	make compress
	# make build
	make push
	make view

push:

	git checkout master
	git add -A 
	git commit -m'updating...'
	git push origin master 
	git checkout gh-pages 
	git merge master 
	git push origin gh-pages
	git checkout master
