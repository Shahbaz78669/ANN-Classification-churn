1. Problem Framing — What am I predicting, and from what?

"Given customer attributes, predict Exited (0/1)." This is binary classification. That single sentence decides your loss function (binary_crossentropy), output activation (sigmoid), and output neurons (1) before you write a line of code.

2. Load & Inspect — pd.read_csv(), .head(), .value_counts()

You're just meeting the data. Which columns are noise (IDs, names)? Which are categorical vs numeric? Which is the target?

3. Clean & Drop Noise

RowNumber, CustomerId, Surname — dropped, because they don't causally relate to churn. This step is really "removing anything that's an identifier, not a signal."

4. Encode Categorical → Numeric

Models only understand numbers.

Binary categories (Gender: Male/Female) → LabelEncoder (0/1)
Multi-class categories (Geography: France/Germany/Spain) → OneHotEncoder (avoids implying false order like France < Germany < Spain)

5. Split, Then Scale

train_test_split FIRST, StandardScaler SECOND — always in that order, to prevent test data "leaking" into your training statistics.
Fit the scaler only on X_train, then transform both.

6. Build → Compile → Train

Build: stack of Dense layers (architecture = how much "thinking capacity" the model has)
Compile: tell it how to learn (optimizer=Adam, loss=binary_crossentropy, metrics=accuracy)
Train: .fit() with callbacks (EarlyStopping to avoid overfitting, TensorBoard to watch progress)

7. Save Everything Needed to Reproduce Predictions

Not just model.h5 — also scaler.pkl, label_encoder_gender.pkl, onehot_encoder_geography.pkl. This is the step beginners forget. A model alone is useless without the exact preprocessing steps that produced its training data — a new raw customer row needs the same encoding/scaling before prediction.

If I were a student rebuilding this from scratch, here's how I'd think

I wouldn't start by writing code. I'd ask myself, in order:

What's the target column, and what type of problem is it? (classification vs regression)
What columns are useless to the model? (IDs, names)
What columns need encoding, and which kind? (binary vs multi-category)
What's my train/test split strategy?
What does my output layer need to look like, given the problem type?
What do I need to persist for someone to use this model later on new data?