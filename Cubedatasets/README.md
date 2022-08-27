BS723 - Introduction to Machine Learning in Architecture: Challenges
Challenge #1:
Problem Definition:
Given a set of prisms with different dimensions and with different colors in
the first dataset, train a model that can classify a new prism to the
corresponding color subset.


Dataset:
The first dataset is composed of 72 different prisms with 8 different colors.
Dataset features that represent the geometry and the color of a prism to train
the model are up to you.


Expected Outcome:
Using at most 18 randomly selected prisms from the test set, classify at least
12 prisms correctly. Also, try to make a confusion matrix of your results in the
test set.



Challenge #2:
Problem Definition:
Given a set of prisms with different dimensions and with different colors in
the second dataset, train a model that can predict the color of a new prism.
Dataset:
The second dataset is composed of another set of 72 different prisms with 72
unique colors. Features to be used as training data to predict the color of a
given prism can be extracted from the given model.
Expected Outcome:
Using at most 12 randomly selected prisms from the test set, predict the color
values of prisms with at most %20 error margin (if the ground-truth value of
the color is 100, your prediction should in between 80-120).
Challenge #3:
Problem Definition:
Given a set of prisms with different dimensions and with different colors in
the third dataset, train a model that can construct a prism when given a new
color.
Dataset:
The third dataset is again composed of a new set of 72 different prisms with
72 different colors. Features of data to be used to construct a prism can also
be used to train the model that predicts the values.
Expected Outcome:
Using at most 9 randomly selected prisms from the test set, predict the color
values of prisms with at most %10 error margin (if you survived the second
challenge, you should know what an error margin means).
