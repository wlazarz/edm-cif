processing_columns = {
    'soybean-large': {
        'outliers': ['charcoal-rot', 'phytophthora-rot', 'downy-mildew',
                     'brown-stem-rot', 'powdery-mildew'],
        'one_elem': ['2-4-d-injury', 'herbicide-injury', 'diaporthe-pod-&-stem-blight', 'cyst-nematode'],
        'big_class': ['alternarialeaf-spot', 'brown-spot', 'phytophthora-rot', 'frog-eye-leaf-spot', 'anthracnose',
                      'rhizoctonia-root-rot', 'purple-seed-stain', 'bacterial-pustule', 'bacterial-blight',
                      'phyllosticta-leaf-spot'],
        'methods': [3, 5, 'all']
    },

    'balance-scale': {
        'outliers': ['B'],
        'one_elem': [],
        'big_class': ['R', 'L'],
        'methods': ['all']
    },
    'car': {
        'outliers': ['good', 'vgood'],
        'one_elem': [],
        'big_class': ['unacc', 'acc', 'good', 'vgood'],
        'methods': [1, 'all']
    },
    'nursery': {
        'big_class': ['priority', 'not_recom', 'very_recom', 'spec_prior'],
        'one_elem': ['recommend'],
        'outliers': ['priority', 'not_recom', 'very_recom'],
        'methods': [1, 2]
    },
    'NPHA-doctor-visits': {
        'big_class': ['3', '2'],
        'one_elem': [],
        'outliers': ['1'],
        'methods': ['all']
    },
    'flare1': {
        'big_class': ['C', 'D', 'B', 'H'],
        'one_elem': [],
        'outliers': ['B', 'F', 'E'],
        'methods': [2, 'all']
    },
    'primary-tumor': {
        'big_class': ['1', '2', '4', '5', '7', '11', '12', '14', '18', '22'],
        'one_elem': ['16', '6', '15', '10', '20', '21'],
        'outliers': ['19', '8', '13', '17', '3'],
        'methods': ['only_outliers', 3, 'all']
    },
    'monks': {
        'big_class': ['0', '1'],
        'one_elem': [],
        'outliers': ['0', '1'],
        'methods': [1]
    },
    'lymphoraphy': {
        'big_class': ['3', '2'],
        'one_elem': ['-1'],
        'outliers': [],
        'methods': [0]
    },
    'splice': {
        'big_class': ['EI', 'IE', 'N'],
        'one_elem': [],
        'outliers': ['EI', 'IE'],
        'methods': [1, 'all']
    },
    'hiv_protease_cleavage': {
        'big_class': ['1'],
        'one_elem': ['-1'],
        'outliers': [],
        'methods': [0]
    },
    'Interests_group': {
        'big_class': ['C', 'P', 'R', 'I'],
        'one_elem': [],
        'outliers': ['C', 'P', 'R', 'I'],
        'methods': [1, 3]
    },
    'bike_buyers': {
        'big_class': ['No', 'Yes'],
        'one_elem': [],
        'outliers': ['No', 'Yes'],
        'methods': [1]
    }
}