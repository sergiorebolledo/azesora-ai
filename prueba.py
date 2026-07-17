from src.agente import consultar_azesora

print("⏳ Consultando a Azesora AI...")
res = consultar_azesora("¿Cómo tributan las ganancias de acciones en EE.UU. desde Chile?")
print("\n🤖 RESPUESTA DE LA IA:")
print(res["respuesta"])
print("\n📁 FUENTES UTILIZADAS:")
print(res["fuentes"])