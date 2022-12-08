import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';


import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AccueilComponent } from './accueil/accueil/accueil.component';
import { ContactComponent } from './contact/contact/contact.component';
import { SendComponent } from './send-receve/send/send.component';
import { ReceveComponent } from './send-receve/receve/receve.component';
import { HeaderComponent } from './header/header.component';
import { FooterComponent } from './footer/footer.component';
import { PageNotFoundComponent } from './page-not-found/page-not-found.component';
import { LoginComponent } from './login/login.component';
import { HttpClientModule } from "@angular/common/http";
import { DeconnexionComponent } from './deconnexion/deconnexion.component';

@NgModule({
  declarations: [
    AppComponent,
    AccueilComponent,
    ContactComponent,
    LoginComponent,
    SendComponent,
    ReceveComponent,
    HeaderComponent,
    FooterComponent,
    PageNotFoundComponent,
    DeconnexionComponent
    
  ],
  imports: [
    BrowserModule,
    HttpClientModule,
    AppRoutingModule,
    FormsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
