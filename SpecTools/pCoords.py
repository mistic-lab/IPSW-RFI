#!/usr/bin/env python3
import pandas as pd
import plotly.graph_objects as go
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv", help="csv file to display")
parser.add_argument("--renderer", default='browser', help="Directory of basebands")
parser.add_argument("--colorKey", default="", help="key of item to usr for color")
parser.add_argument("--colorscale", default="rdylbu", help="colorscale to use if colorKey is set")
args = parser.parse_args()

df = pd.read_csv(args.csv)

coords = []

for i in df.keys():
    if i == args.colorKey:
        continue
    coords.append(dict(label=i, values=df[i]))

if args.colorKey == "":
    lineParams = dict( color='black', showscale=False)
    title = ""
else:
    lineParams = dict( color=df[args.colorKey], showscale=True, colorscale=args.colorscale)
    title = "colored by {}".format(args.colorKey)

fig = go.Figure(
    data = go.Parcoords(
        line = lineParams,
        dimensions = coords,
    ),
    layout=go.Layout(
        title=go.layout.Title(text=title)
    )
)

fig.update_layout(
    plot_bgcolor = 'grey',
    paper_bgcolor = 'grey'
)

fig.show(renderer=args.renderer)