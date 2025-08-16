"""
Sample data fixtures for testing.
"""

# Sample Japanese texts for testing different scenarios
SAMPLE_TEXTS = {
    'simple': "田中太郎は東京でトヨタの車を購入しました。",
    
    'complex': """2024年11月15日、東京国際フォーラムで開催されたAI技術カンファレンスにおいて、
OpenAI社のCEOサム・アルトマン氏が基調講演を行いました。同イベントには、トヨタ自動車株式会社の
豊田章男会長が参加しました。会場では、最新のGPT-5モデルが紹介されました。""",
    
    'no_entities': "これは普通の文章です。特別な固有表現は含まれていません。",
    
    'mixed_entities': """経済産業省の田中智子局長は、自由民主党のデジタル推進委員会で、
東京大学と京都大学の研究について議論しました。ChatGPTとClaudeの性能比較も行われました。""",
    
    'organizations': """株式会社ABEJA、ソフトバンクグループ株式会社、マイクロソフト日本法人、
国立研究開発法人海洋研究開発機構が参加しました。""",
    
    'locations': "東京、大阪、六本木、関西地方、奈良県で開催されます。",
    
    'products': "iPhone、GPT-4、Google Bard、Apple Intelligence、ChatGPT Plus。",
    
    'events': "AI技術カンファレンス、関西AI展、デジタル推進委員会、成果報告会。"
}

# Expected entity results for validation
EXPECTED_ENTITIES = {
    'simple': [
        {'word': '田中太郎', 'entity_type': 'PER'},
        {'word': '東京', 'entity_type': 'LOC'},
        {'word': 'トヨタ', 'entity_type': 'ORG'}
    ],
    
    'mixed_entities': [
        {'word': '経済産業省', 'entity_type': 'ORG-O'},
        {'word': '田中智子', 'entity_type': 'PER'},
        {'word': '自由民主党', 'entity_type': 'ORG-P'},
        {'word': '東京大学', 'entity_type': 'INS'},
        {'word': '京都大学', 'entity_type': 'INS'},
        {'word': 'ChatGPT', 'entity_type': 'PRD'},
        {'word': 'Claude', 'entity_type': 'PRD'}
    ]
}

# Mock NER pipeline results
MOCK_NER_RESULTS = {
    'simple': [
        {
            'word': '田中太郎',
            'entity_group': 'PER',
            'score': 0.9999,
            'start': 0,
            'end': 3
        },
        {
            'word': '東京',
            'entity_group': 'LOC',
            'score': 0.9998,
            'start': 4,
            'end': 6
        },
        {
            'word': 'トヨタ',
            'entity_group': 'ORG',
            'score': 0.9997,
            'start': 7,
            'end': 10
        }
    ]
}

# Sample analysis results
SAMPLE_ANALYSIS_RESULTS = [
    {
        'filename': 'doc1.txt',
        'content': SAMPLE_TEXTS['simple'],
        'entities': [
            {'word': '田中太郎', 'entity_type': 'PER', 'score': 0.99, 'start': 0, 'end': 3},
            {'word': '東京', 'entity_type': 'LOC', 'score': 0.98, 'start': 4, 'end': 6}
        ],
        'entity_count': 2,
        'analysis_time': '2024-01-01T10:00:00'
    },
    {
        'filename': 'doc2.txt',
        'content': SAMPLE_TEXTS['organizations'],
        'entities': [
            {'word': 'ABEJA', 'entity_type': 'ORG', 'score': 0.97, 'start': 3, 'end': 8},
            {'word': 'ソフトバンクグループ', 'entity_type': 'ORG', 'score': 0.96, 'start': 10, 'end': 19}
        ],
        'entity_count': 2,
        'analysis_time': '2024-01-01T10:01:00'
    }
]

# Sample statistics
SAMPLE_STATISTICS = {
    'total_documents': 2,
    'total_entities': 4,
    'avg_entities_per_doc': 2.0,
    'entity_type_counts': {'PER': 1, 'LOC': 1, 'ORG': 2},
    'entity_word_counts': {'田中太郎': 1, '東京': 1, 'ABEJA': 1, 'ソフトバンクグループ': 1},
    'most_common_entities': [('田中太郎', 1), ('東京', 1), ('ABEJA', 1)],
    'entity_type_distribution': {'PER': 25.0, 'LOC': 25.0, 'ORG': 50.0},
    'documents_stats': [
        {
            'filename': 'doc1.txt',
            'entity_count': 2,
            'unique_entity_types': 2,
            'text_length': len(SAMPLE_TEXTS['simple'])
        },
        {
            'filename': 'doc2.txt',
            'entity_count': 2,
            'unique_entity_types': 1,
            'text_length': len(SAMPLE_TEXTS['organizations'])
        }
    ]
}

# Entity descriptions for testing
ENTITY_DESCRIPTIONS = {
    'PER': '人名',
    'ORG': '一般企業・組織',
    'ORG-P': '政治組織',
    'ORG-O': 'その他の組織',
    'LOC': '場所・地名',
    'INS': '施設・機関',
    'PRD': '製品',
    'EVT': 'イベント'
}