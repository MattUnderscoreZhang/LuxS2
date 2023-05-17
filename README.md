# Lux AI Season 2 Entry

This is my entry for the Kaggle Lux AI Season 2 competition (https://www.kaggle.com/competitions/lux-ai-season-2/). I didn't finish in time, but I still learned a lot from the process!

This codebase uses a curriculum-learning approach, where the decision-making agent is given lessons one at a time, and is allowed to progress once the loss is low enough for each lesson. The agent uses the following decision-making process:

1. Observations are calculated for each space on the board. These observations are then fed into a neural net which appends local features calculated via convolutions around the immediate neighborhood of each space, and global features based on an analysis of the entire board.
2. These extended observations are then used to assign a separate role to each position on the board. Roles include mining, carrying payloads, sabatage, etc.
3. The agent calculates an action for each space on the board, using a neural net corresponding to the role assigned to that space.
4. Actions are fed into the Lux environment, and the agent is updated via the PPO algorithm.
