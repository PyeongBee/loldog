import { useMemo, useState } from "react";
import { View, Text, StyleSheet, Pressable, FlatList } from "react-native";
import { useRouter } from "expo-router";

type Champion = {
  id: string;
  name: string;
  image?: string;
};

const CHAMPIONS: Champion[] = [
  // { id: "ashe", name: "Ashe" },
  // { id: "ahri", name: "Ahri" },
  { id: "lee-sin", name: "Lee Sin" },
  // { id: "lux", name: "Lux" },
  // { id: "jinx", name: "Jinx" },
  // { id: "zed", name: "Zed" },
  { id: "vayne", name: "Vayne", image: require("../assets/images/vayne.jpg") },
];

export default function HomeScreen() {
  const router = useRouter();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const selectedChampion = useMemo(() => CHAMPIONS.find((c) => c.id === selectedId), [selectedId]);

  const handleStart = () => {
    if (!selectedChampion) return;
    router.push({
      pathname: "/detect",
      params: { champion: selectedChampion.name, championId: selectedChampion.id },
    });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>챔피언 선택</Text>
      <Text style={styles.subtitle}>미니맵에서 감지할 챔피언을 선택하세요.</Text>

      <FlatList
        data={CHAMPIONS}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.list}
        numColumns={2}
        columnWrapperStyle={styles.column}
        renderItem={({ item }) => {
          const active = item.id === selectedId;
          return (
            <Pressable
              onPress={() => setSelectedId(item.id)}
              style={[styles.card, active && styles.cardActive]}
            >
              <Text style={styles.cardName}>{item.name}</Text>
              <Text style={styles.cardHint}>{active ? "선택됨" : "탭하여 선택"}</Text>
            </Pressable>
          );
        }}
      />

      <Pressable
        onPress={handleStart}
        disabled={!selectedChampion}
        style={[styles.primaryButton, !selectedChampion && styles.primaryButtonDisabled]}
      >
        <Text style={styles.primaryLabel}>
          {selectedChampion ? `${selectedChampion.name} 감지 시작` : "챔피언을 먼저 선택"}
        </Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 48,
    paddingHorizontal: 20,
    backgroundColor: "#0b1115",
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: "#fff",
    marginBottom: 6,
  },
  subtitle: {
    fontSize: 15,
    color: "#c6d0d9",
    marginBottom: 16,
  },
  list: {
    paddingVertical: 12,
    gap: 12,
  },
  column: {
    gap: 12,
  },
  card: {
    flex: 1,
    minHeight: 96,
    backgroundColor: "#1b2732",
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: "#243241",
    justifyContent: "space-between",
  },
  cardActive: {
    borderColor: "#ff4d4f",
    backgroundColor: "#212f3b",
  },
  cardName: {
    fontSize: 18,
    fontWeight: "700",
    color: "#f8fbff",
  },
  cardHint: {
    fontSize: 13,
    color: "#9fb0c2",
  },
  primaryButton: {
    marginTop: "auto",
    paddingVertical: 16,
    borderRadius: 14,
    alignItems: "center",
    backgroundColor: "#ff4d4f",
    marginBottom: 48,
  },
  primaryButtonDisabled: {
    backgroundColor: "#3a4756",
  },
  primaryLabel: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
  },
});
