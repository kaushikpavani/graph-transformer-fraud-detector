import gtfd
from gtfd.config import GTFDConfig
from gtfd.data.generator import SyntheticTransactionGenerator
from gtfd.graph.builder import PaymentGraphBuilder
from gtfd.model.gtfd import GTFDModel

print('GTFD version:', gtfd.__version__)
print('Imports successful!')

# Quick test
config = GTFDConfig()
config.data.num_accounts = 50
config.data.num_merchants = 10
config.data.num_transactions = 500
config.data.fraud_rate = 0.05

generator = SyntheticTransactionGenerator(config.data)
df, rings = generator.generate_dataset()
print(f'Generated {len(df)} transactions, {len(rings)} rings')

builder = PaymentGraphBuilder()
data = builder.build_graph(df)
print(f'Graph: {data.num_nodes} nodes, {data.num_edges} edges')

model = GTFDModel(config.model)
model.add_fraud_head(num_classes=2)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
print('All basic tests passed!')