# Steps

## **Step 1: Generate Semantic Features for Words**

- Prompt GPT to generate semantic features for each word based on previous publications.

<aside>
💡 *We are conducting an investigation to understand how people interpret words for their meaning. In order to assist us in this research, we require information regarding the knowledge individuals possess about various concepts in the world. You will be provided with a list of words, each representing a specific concept. Your task is to come up with at least 12 properties related to the concept represented by the word. These properties can encompass a range of aspects, including physical characteristics such as internal and external features, sensory attributes (how it looks, sounds, smells, feels, or tastes), functional traits (its purpose, usage, when and where it is used, and by whom), associations or categories it belongs to, behavioral traits, and its origin. Please try to make the answer concise. Also, please start with more general characteristics and progress to more specific details.*

*Please note that, even if some of the words can be interpreted as something other than a noun (e.g., 'camp' could refer to the place where a tent is pitched or the act of camping), all the words listed should be treated as nouns only (e.g., 'camp,' the place).*

*Here are three examples of the word and its possible properties:*

*Duck
Is a bird
Is an animal
Waddles
Flies
Migrates
Lays eggs
Has webbed feet
Has feathers
Lives in ponds
Lives in water
Hunted by people
Is edible*

*Cucumber
Is a vegetable
Has green skin
Has a white inside
Has seeds inside
Is long
Is cylindrical
Grows on vines
Is edible
Is crunchy
Used for making pickles
Eaten in salads*

*Stove
Is an appliance
Produces heat
Has elements
Made of metal
Is hot
Is electrical
Runs on wood
Runs on gas
Found in kitchens
Used for baking
Used for cooking food*

*While you may think of additional or different types of properties for these concepts, these examples serve as a guide for what we are seeking.*

</aside>

## **Step 2: Obtain Embeddings for the Generated Features**

- Utilize sentence embeddings since the generated features can be phrases or sentences.
- Sentence embeddings enable grouping similar features together (see Step 3).

## **Step 3: Cluster and Label the Features**

- Apply K-means clustering for its efficiency, with a selected value of K=1000.
- Utilize TF-IDF (Term Frequency-Inverse Document Frequency) to identify keywords (features) for each cluster.
- Transform the data into a sparse word-feature matrix, where the first column represents words, and the subsequent columns represent unique features. If a word possesses a specific feature, it will be encoded as 1; otherwise, it will be encoded as 0.

## **Step 4: Validate the Binary Feature Representation**

- Calculate pairwise similarity between all words based on the binary representation.
- Calculate pairwise similarity between all words using word2vec representation.
- Evaluate the correlation between the two pairwise similarity vectors for significant correlations.
