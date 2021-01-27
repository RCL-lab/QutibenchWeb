# Auto-convert Jupyter Notebooks To Posts

[`fastpages`](https://github.com/fastai/fastpages) will automatically convert [Jupyter](https://jupyter.org/) Notebooks saved into this directory as blog posts!

You must save your notebook with the naming convention `YYYY-MM-DD-*.ipynb`.  Examples of valid filenames are:

```shell
2020-01-28-My-First-Post.ipynb
2012-09-12-how-to-write-a-blog.ipynb
```

If you fail to name your file correctly, `fastpages` will automatically attempt to fix the problem by prepending the last modified date of your notebook. However, it is recommended that you name your files properly yourself for more transparency.

See [Writing Blog Posts With Jupyter](https://github.com/fastai/fastpages#writing-blog-posts-with-jupyter) for more details.


### Adding New Measurements
	If you want to add new measurements you must add the measurements to the file: 
		- data/cleaned_csv/backup.csv
			This csv file owns all data from all measurements that is processed in the website.
			Be careful with the names of the platforms/CNNs. They need to be the same as in other csv files because they are linked.
			To add new measurements you can use a handy script that fills up 3 columns of that the backup.csv file automatically, which is:
				-scripts/script_add_columns.py
					-please verify if all data (throughput values, etc...) in script_add_columns.py is correctly filled up, and make all modifications necessary.
			


### Website structure
	-Most code to generate plots resides in 3 scripts:
		-overlapped_pareto.py : this script owns all methods to create the pareto plots, the overlapped pareto plots and efficiency bar charts
		-rooflines_heatmaps.py : contains all methods to create the rooflines and heatmaps
		-util.py : contains some utility methods. 
		
	-FAIR data documentation resides in data/FAIR_data
	