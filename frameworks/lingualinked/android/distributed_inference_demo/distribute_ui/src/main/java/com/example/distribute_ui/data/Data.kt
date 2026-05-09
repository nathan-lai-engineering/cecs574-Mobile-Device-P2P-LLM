package com.example.distribute_ui.data
import com.example.distribute_ui.ui.Messaging

val initialMessages = mutableListOf(
    Messaging(
        "Robot",
        "Test",
        "03:07 pm",
        null
    ),
    Messaging(
        "Me",
        "Test Reply",
        "03:07 pm",
        null
    )
)

val exampleModelName = listOf(
        "tinyllama"
)

val modelMap: HashMap<String, String> = hashMapOf(
    "tinyllama" to "tinyllama"
)
