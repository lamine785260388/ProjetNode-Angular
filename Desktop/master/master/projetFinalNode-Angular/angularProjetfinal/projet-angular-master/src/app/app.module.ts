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
import { ResumetransactionComponent } from './send-receve/resumetransaction/resumetransaction.component';
import { ListTransactionComponent } from './Transaction/list-transaction/list-transaction.component';
import { CreateuserComponent } from './administrateur/createuser/createuser.component';
import { ListeUserComponent } from './administrateur/liste-user/liste-user.component';
import { CreerAgenceComponent } from './creer-agence/creer-agence.component';
import { CreerComponent } from './SousAgence/creer/creer.component';


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
    DeconnexionComponent,
    ResumetransactionComponent,
    ListTransactionComponent,
    CreateuserComponent,
    ListeUserComponent,
    CreerAgenceComponent,
    CreerComponent,


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
