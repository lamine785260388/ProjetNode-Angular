import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AccueilComponent } from './accueil/accueil/accueil.component';
import { ContactComponent } from './contact/contact/contact.component';
import { LoginComponent } from './login/login.component';
import { PageNotFoundComponent } from './page-not-found/page-not-found.component';
import { SendComponent } from './send-receve/send/send.component';
import { DeconnexionComponent } from './deconnexion/deconnexion.component';
import { ReceveComponent } from './send-receve/receve/receve.component';
import { ListTransactionComponent } from './Transaction/list-transaction/list-transaction.component';

const routes: Routes = [
  {path: '', component:AccueilComponent},
  {path:'contact', component:ContactComponent},
  {path:'login', component:LoginComponent},
  {path:'send', component:SendComponent},
  {path:'deconnexion',component:DeconnexionComponent},
  {path:'recevoir',component:ReceveComponent},
  {path:'listTransaction',component:ListTransactionComponent},

  {path:'**', component:PageNotFoundComponent}
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
