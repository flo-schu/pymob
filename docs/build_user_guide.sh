#!/bin/usr/bash

# gettig started
jupyter nbconvert --to markdown --execute docs/source/user_guide/superquickstart.ipynb

# user guide
# quickstaart needs to go before framework overview because generated scenario is needed for the framework overview
jupyter nbconvert --to markdown --execute docs/source/user_guide/quickstart.ipynb
jupyter nbconvert --to markdown --execute docs/source/user_guide/Introduction.ipynb
jupyter nbconvert --to markdown --execute docs/source/user_guide/advanced_tutorial_ODE_system.ipynb
jupyter nbconvert --to markdown --execute docs/source/user_guide/framework_overview.ipynb
