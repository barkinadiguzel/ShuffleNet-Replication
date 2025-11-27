# Number of groups in pointwise group convolution
GROUPS = 3  # g = 1, 2, 3, 4, 8 

# Stage output channels for different group numbers (
# Format: [stage2, stage3, stage4]
STAGE_OUT_CHANNELS = {
    1: [144, 288, 576],
    2: [200, 400, 800],
    3: [240, 480, 960],
    4: [272, 544, 1088],
    8: [384, 768, 1536]
}

# Number of ShuffleNet units per stage 
STAGE_REPEATS = [2, 2, 2]  # Stage2, Stage3, Stage4

# Number of classes for classification
NUM_CLASSES = 1000
