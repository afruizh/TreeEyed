CONFIG_MAX_TILE_PROCESSING = 25

CONFIG_DEBUG = True


def config_debug(*msg):    
    if (CONFIG_DEBUG):
        print(*msg)