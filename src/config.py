# Number of groups in pointwise group convolution
GROUPS = 3  # g = 1, 2, 3, 4, 8 

# Width multiplier for controlling model size
WIDTH_MULTIPLIER = 1.0  # 0.5, 1.0, 1.5, 2.0 

# Stage output channels for g=1,2,3,4,8 (from the paper)
# Format: [stage2, stage3, stage4]
STAGE_OUT_CHANNELS = {
    1: [144, 288, 576],
    2: [200, 400, 800],
    3: [240, 480, 960],
    4: [272, 544, 1088],
    8: [384, 768, 1536]
}

# Number of ShuffleNet units per stage (repeat)
STAGE_REPEATS = [4, 8, 4]  # Example: Stage2-4 repeat counts

# Number of classes for classification
NUM_CLASSES = 1000
