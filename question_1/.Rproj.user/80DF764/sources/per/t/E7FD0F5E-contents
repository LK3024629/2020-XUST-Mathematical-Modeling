library(shiny)

ui <-fluidPage(pageWithSidebar(
  headerPanel("This Shiny apps calculate your Age and display your Zodiac sign"),
  sidebarPanel(
    dateInput("birthday", "Please enter your birthday (YYYY-MM-DD) :"),
    checkboxGroupInput("display", "Do you want to:",
                       c("Display Age" = "1",
                         "Display your Zodiac" = "2")),
    # h3("Function:"),
    h5("This application calculates your age and displays your zodiac sign"),
    h5("You may choose what to display by clicking any of the checkboxes.")
  ),
  mainPanel(
    h5('Your Birthday is:'),
    verbatimTextOutput("age"),
    verbatimTextOutput("zod")
  )
))


server <- function(input, output) {
  output$age <- renderPrint({format(input$birthday,"%A %d %B %Y")})
  output$zod <- renderPrint(ProcessOptions({input$display}))
  ProcessOptions <- function(x){
    input <- "Please choose to calculate your age or display your zodiac sign:"
    if (length(x)==0){
      return(cat(paste(input,"You have chosen nothing!",sep="\n")))
    }
    else if (length(x)>0){
      for (i in 1:length(x)){
        input <- paste(input,showOption(x[i]),sep="\n")
      }
      return(cat(input))
    }   
  }
  showOption <- function(i){
    if (i==1){
      x1 <- "\n-Show your age now -->"
      date1 <- strptime(input$birthday, format="%d.%m.%Y")
      date2 <- strptime(date(), format="%d.%m.%Y")
      Days <- difftime(Sys.Date(),input$birthday,units="days")
      Days <- as.numeric(Days)
      calDays <- floor((Days %% 365.25) %% 30.4)
      calAge <- paste(calDays, "days old",sep=" ")
      Days <- Days - calDays
      calMonths <- floor((Days %% 365.25) / 30.4)
      calAge <- paste(paste(calMonths, "months and",sep=" "),calAge,sep=" ")
      Days <- Days - (calMonths*30.4)
      calYears <- floor(Days / 365.25)
      calAge <- paste(paste(calYears, "years and",sep=" "),calAge,sep=" ")
      finalAge <- paste("You are now",calAge, sep=" ")
      if (Days<1){return(paste(x1,"You are less than 1 year old",sep="\n"))}
      return(paste(x1,finalAge,sep="\n"))
    }
    else if (i==2){
      x1 <- "\n-Show your Zodiac Sign -->\n"
      x <- as.numeric(paste0(format(input$birthday,"%m"),format(input$birthday,"%d")))
      hor <- function(x){if (x>=320 && x<420) return('Aries') 
        else if (x>=420 && x<521) return('Taurus')
        else if (x>=521 && x<621) return('Gemini')
        else if (x>=621 && x<723) return('Cancer')
        else if (x>=723 && x<823) return('Leo')
        else if (x>=823 && x<923) return('Virgo')
        else if (x>=923 && x<1023) return('Libra')
        else if (x>=1023 && x<1122) return('Scorpio')
        else if (x>=1122 && x<1222) return('Sagittarius')
        else if (x>=1222 || x<120) return('Capricorn')
        else if (x>=120 && x<218) return('Aquarius')
        else if (x>=218 && x<320) return('Pisces')
      }
      zodiac <- paste(x1, paste("Your zodiac is ",hor(x),sep=""),sep="")
      return(zodiac)
    }
  }
}


shinyApp(ui = ui, server = server)
