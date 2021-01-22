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
	This file owns all data that is processed in the website.
	To add new measurements you can use a handy script that fills up 3 columns of that csv file automatically, which is:
		-scripts/script_add_columns
		-please verify of all data in that script is correctly filled up.


### Website structure
	-Most code to generate plots resides in 3 scripts:
		-overlapped_pareto.py : this script owns all methods to create the pareto plots and the overlapped pareto plots
		-rooflines_heatmaps.py : containsa ll methods to create heatmaps
		-util.py : contains utility methods