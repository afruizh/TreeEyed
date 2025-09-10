CONFIG_MAX_TILE_PROCESSING = 2500

CONFIG_DEBUG = True


def config_debug(*msg):    
    if (CONFIG_DEBUG):
        print(*msg)