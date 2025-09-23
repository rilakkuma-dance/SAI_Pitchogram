import panphon
import panphon.distance

ft = panphon.FeatureTable()

dist = panphon.distance.Distance()
max_dist = 24 * max(4, 4)
example = dist.phoneme_error_rate(["ʃ","e","ʃ","e"], ["x","ie","x","ie"])
similarity = (1 - example / max_dist) * 100

print(f"Example distance: {example}, similarity: {similarity:.2f}%")
print(f"Similarity: {similarity:.2f}%")