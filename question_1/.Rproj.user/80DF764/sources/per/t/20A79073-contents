library(shiny) 

ui<- fluidPage(
  titlePanel("Power Two Calculator"),
  sidebarLayout (
    sidebarPanel(
      numericInput(inputId = "number", label = 
    "Input an integer from 1 to 100", value = 10, min = 1, max=100, step=1),
    submitButton("Tekan")
    ),
    mainPanel(
      h2("Results of Calculation"),
      h3("You Entered"),
      verbatimTextOutput("num"),
      h3("The power two of the input number is:"),
      verbatimTextOutput("power"),
      h3("The power three of the input number is:"),
      verbatimTextOutput("power1")
      
    )
  )
) 
powerTwo <- function (nombor) {nombor*nombor}
powerThree <- function (nombor) {nombor*nombor*nombor}

server <- function(input,output) {
  output$num <- renderPrint(input$number)
  output$power<- renderPrint({powerTwo(input$number)})
  output$power1 <- renderPrint({powerThree(input$number)})
}

shinyApp(ui=ui, server=server) 