# Етап збірки: базовий образ з Java 21 JDK для компіляції
FROM eclipse-temurin:21-jdk-alpine AS build
# Робоча директорія всередині контейнера
WORKDIR /app
# Встановлення Maven для збірки проекту
RUN apk add --no-cache maven
# Копіювання файлу залежностей (окремо для кешування)
COPY pom.xml .
# Завантаження всіх залежностей проекту
RUN mvn dependency:go-offline
# Копіювання вихідного коду
COPY src ./src
# Компіляція проекту в JAR файл без запуску тестів
RUN mvn clean package -DskipTests

# Фінальний етап: легкий образ з JRE для запуску
FROM eclipse-temurin:21-jre-alpine
# Робоча директорія у фінальному образі
WORKDIR /app
# Копіювання скомпільованого JAR з етапу збірки
COPY --from=build /app/target/cnn-evolution-1.0-SNAPSHOT.jar app.jar
# Копіювання даних MNIST
COPY data ./data
# Копіювання директорії для логів
COPY logs ./logs
# Створення точки монтування для логів
VOLUME /app/logs
# Команда запуску головного класу при старті контейнера
ENTRYPOINT ["java", "-cp", "app.jar", "natanius.thesis.cnn.evolution.Evolution"]
