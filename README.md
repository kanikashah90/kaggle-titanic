# kaggle-titanic

## Target of the problem is to predict given a feature set can I determine if that feature set would have died if it boarded titanic or would it have survived.

For each passenger in the test set, you must predict whether or not they survived the sinking ( 0 for deceased, 1 for survived ).  Your score is the percentage of passengers you correctly predict.

The feature names and the type of values that they have:
1. PassengerId -- 1 --> 891
2. Survived -- Either 0 or 1
3. Pclass -- Values 1, 2 or 3
4. Name -- String
5. Sex -- male or female
6. Age -- Some of the feature tuple are missing the age feature. Also, it is a number representing the age of the boarder. Some of the feature have age in decimal. Rethink over it.
7. SibSp -- Following values: 0,1,2,3,4,5,8
8. Parch -- Following value: 0,1,2,3,4,5,6
9. Ticket -- It is a string which has a ticket number on it. Some are just numbers and some are alphanumeric.
10. Fare -- It is a floating point number, most likely the fare for the ticket the person purchased.
11. Cabin -- It is an alphanumeric number. Most of the entries are missing this value and some have a sequence of alphanumeric values as the value.
12. Embarked -- One of the value: Q, S, C. Some of the entries are missing this value.

Learn some relationships between the features and the output. Try to figure out these relationships:
1. See if there is any relationship between Ticket and Fare.
2. See if there is any relationship between Fare and Cabin.
3. See the correlation between Survived and Sex.
4. See the correlation between Survived, Sex and PClass.
5. See if age has any influence on the Survived.
6. See if a SibSp has any influence on the Survived.
7. How is Embarked related to the survival of the passenger.
8. Does the Parch plays any role.

For initial pick on the features, lets choose these one's:
Sex, Pclass, Fare(Based on the bins that I created), SibSp, Parch