---
title: 'Write title here'
tags:
  - python
  - AI
  - other tags
authors:
  - name: Author With ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID^[co-first author] # note this makes a footnote saying 'co-first author'
    affiliation: 2
  - name: Author With No Affiliation^[corresponding author]
    affiliation: 3
affiliations:
 - name: Name Surname, Jr. Fellow, Institution Name
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
   index: 3
date: dd Month yyyy
bibliography: paper.bib
---

# Summary

Summary goes here.

# Statement of need

Statement of need goes here.

# State of the field

State of the field goes here.

# Key Features

Key features go here.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](https://user-images.githubusercontent.com/3244249/151994514-b584b984-a148-4ade-80ee-0f88b0aefa45.png){ width=20% }

# Acknowledgements

Acks go here.

# References

@bibname
