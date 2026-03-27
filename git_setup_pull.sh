#!/bin/bash
# initialize git repository
# git init
# git config user.name "hellboj"
# git config user.email "hellboj@centrum.cz"
# git add .

# setup remote repository
git remote add origin https://github.com/strakrob/ctlbm.git

# pull from remote repository
git pull origin master --allow-unrelated-histories
