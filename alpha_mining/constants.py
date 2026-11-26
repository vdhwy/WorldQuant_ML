# Region
REGIONS = ['USA', 'GLB', 'EUR', 'ASI', 'CHN', 'KOR', 'TWN', 'IND', 'MEA']
# Universes
UNIVERSES = [
    {
        "region": "USA",
        "name": ["TOP3000", "TOP1000", "TOP2000", "TOP500", "TOP200", "ILLIQUID_MINIVOL1M", 'TOPSP500'],
        'delay': [0,1]
    },
    {
        "region": "CHN",
        "name": "TOP2000U",
        'delay': [0,1]
    },
    {
        "region": "ASI",
        "name": ["TOP500", "MINVOL1M", 'MINVOL10M', 'ILLIQUID_MINVOL1M'], 
        'delay': 1
    },
    {
        "region": "EUR",
        "name": ["TOP2500", "TOP1200", "TOPCS1600", "TOP400", 'TOP800', 'ILLIQUID_MINVOL1M'],
        'delay': [0,1]
    },
    {
        "region": "GLB",
        "name": ['TOP3000', 'MINVOL1M', 'MINVOL10M', 'TOPDIV3000'],
        'delay': 1
    },
    {
        "region": "KOR",
        "name": "TOP600",
        'delay': 1
    },
    {
        'region': 'TWN',
        'name': ['TOP500', 'TOP100'],
        'delay': 1
    },
    {
        'region': 'IND',
        'name': 'TOP500',
        'delay': 1
    },
    {
        'region': 'MEA',
        'name': 'TOP400',
        'delay': 1
    }
]