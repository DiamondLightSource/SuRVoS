---
title: Contribute to SuRVoS
layout: documentation
toc-section: 5
toc-title: Contribute
---

SuRVoS is an Open Source workbench for semi-automatic segmentation of difficult and large volumes. There are several ways to contribute to SuRVoS, which all of them require a GitHub Account:

1. Use it and report issues, bugs and potential features that would benefit SuRVoS!

    [https://GitHub.com/DiamondLightSource/SuRVoS/issues](https://GitHub.com/DiamondLightSource/SuRVoS/issues)

2. If you are a developer, the source code of SuRVoS is publicy available. Feel free to have a look at it, find and fix bugs, add new features or rework some areas of the tool and submit them as a Pull Requests (PR).

3. Similarly, this web-based documentation is also publicy available in GitHub (`gh-pages` branch). New content for the documentation, correction typos or new tutorials in the form of PR are very welcome.

For the points 2. and 3. we would recommend reading before the [Collaborating with issues and pull requests](https://help.GitHub.com/categories/collaborating-with-issues-and-pull-requests/) section in GitHub.

## Contributing to the Source Code

### 1. For the first time:

- Go to the GitHub page [https://GitHub.com/DiamondLightSource/SuRVoS](https://GitHub.com/DiamondLightSource/SuRVoS) and click on the top right button **Fork** to create your own copy of the repository. **NOTE:** You only have to do this once for both code and documentation.

- On your computer, open a terminal and move to a folder where you want to store the repository and type:

      $> git clone -b master --single-branch https://GitHub.com/your-username/SuRVoS.git SuRVoS

    replacing *your-username* with your GitHub username.


- Then `cd SuRVoS` to navigate to the appropiate folder and add the upstream repository:

      $> git remote add upstream https://GitHub.com/DiamondLightSource/SuRVoS.git

### 2. Make changes to the documentation:

- Before you start, pull the latest changes from the main repository:

      $> git pull upstream master

- Create a branch for the feature you want to work on with a new name, such as `code_changes`:

      $> git checkout -b code_changes master

- Commit locally your changes periodically by using `git add` and `git commit` commands.

### 3. Push the changes to the main SuRVoS repository:

- Once you are happy with the current status of your branch, push changes back to your GitHub:

      $> git push origin doc_changes

    remember to replace *doc_changes* with the name of your branch.

- Go to your repository on GitHub. The new branch will show up with a green Pull Request button - click on it and submit your changes.

## Contributing to the web-based Documentation

### For the first time:

- Go to the GitHub page [https://GitHub.com/DiamondLightSource/SuRVoS](https://GitHub.com/DiamondLightSource/SuRVoS) and click on the top right button **Fork** to create your own copy of the repository. **NOTE:** You only have to do this once for both code and documentation

- On your computer, open a terminal and move to a folder where you want to store the repository and type:

      $> git clone -b gh-pages --single-branch https://GitHub.com/your-username/SuRVoS.git SuRVoS_DOC

    replacing *your-username* with your GitHub username.

- Then `cd SuRVoS_DOC` to navigate to the appropiate folder and add the upstream repository:

      $> git remote add upstream https://GitHub.com/DiamondLightSource/SuRVoS.git

### Make changes to the documentation:

- Before you start, pull the latest changes from the main repository:

      $> git pull upstream gh-pages

- Create a branch for the feature you want to work on with a new name, such as `doc_changes`:

      $> git checkout -b doc_changes gh-pages

- Documentation pages are stored in the `/docs/` folder and images in `/images/` folder. Documentation is writen in Markdown format which is then converted to a web-based content automatically by means of [Jekyll](https://jekyllrb.com/).

- To modify the content of the documentation just add a new file in the `/docs/` folder or edit the content of an existing one. New images can be put into `/images/` folder and linked in the text using markdown notation:

      ![alt]({{ "{{site.baseurl" }}}}/images/path_to_image.png)

    By default the image will fill the whole width (or its maximum size). It can be aligned with the text by setting `alt` to either of the following,

      left, left30, left40, left50, left60, left70, left80, left90
      right, right30, right40, right50, right60, right70, right80, right90

    which will align the image to the left or right respectively and set the width to the percentage in the name. As an example, the following line will align the image to the right and fill 40% of the page width:

      ![right40]({{ "{{site.baseurl" }}}}/images/path_to_image.png)

- Commit locally your changes periodically by using `git add` and `git commit` commands.

      git add [changed_files_separated_by_comma]          # For few files
      git add -A                                          # Add ALL changed files

    And commit changes:

      $> git commit -am 'commit_message'

    and replace *commit_message* with some description of the changes, as it is what will be registered in logs.

### Preview changes locally in your computer:

- This step is totally optional and only required if you have made major changes and want to make sure everything is going to be visualized appropiately. Skip to step 4 if not needed.

- Follow installation instructions for Jekyll in the official website:

    - Windows: [https://jekyllrb.com/docs/windows/#installation](https://jekyllrb.com/docs/windows/#installation)

    - Linux/Mac: [https://jekyllrb.com/docs/installation/](https://jekyllrb.com/docs/installation/)

- On the root of the repository (folder `SuRVoS_DOC/`) run the following command to serve the webpage:

      $> jekyll serve --watch

- The web-based doumentation will be locally visible in your computer by entering to the following URL:

    [http://localhost:4000/SuRVoS/](http://localhost:4000/SuRVoS/)

    Any changes made to the documentation will be automatically visible in that URL after refreshing the pages (F5 in a web browser).

### Push the changes to the main SuRVoS repository:

- Once you are happy with the current status of your branch, push changes back to your GitHub:

      $> git push origin doc_changes

    remember to replace *doc_changes* with the name of your branch.

- Go to your repository on GitHub. The new branch will show up with a green Pull Request button - click on it and submit your changes.
