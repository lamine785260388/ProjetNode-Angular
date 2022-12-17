import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import { Router } from '@angular/router';
import { Transaction } from 'src/app/class/transaction';
import Swal from 'sweetalert2';

@Component({
  selector: 'app-receve',
  templateUrl: './receve.component.html',
  styleUrls: ['./receve.component.css']
})
export class ReceveComponent {
  donne:Transaction|any
  constructor(private router:Router,private http:HttpClient){
    if(sessionStorage.getItem('isloggin')!='true'){
     sessionStorage.setItem('url','recevoir')
     this.router.navigate(['login'])
    }
  }
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };

  submit (form: NgForm) {
console.log('suis la')
//recupération donné paiment
      
let codepaiement = form.value.CodePaiement
let numeroDepiece = form.value.numeroDepiece
let nomRecepteur = form.value.nomRecepteur

console.log(codepaiement,numeroDepiece,nomRecepteur);
this.http
.post<Transaction|any>(
  "http://localhost:3000/api/InsertPaiement",{id:codepaiement,recepteurid:numeroDepiece,nom_recepteur:nomRecepteur,idsousAgence:sessionStorage.getItem('sousAgenceid')},
  this.httpOptions
  )
.subscribe(res=>{
  if(res.erreur=='false'){
    this.donne=res.data
    if(this.donne.DeviceDest!=this.donne.DeviceOrigine){
      if(this.donne.DeviceDest!="Francs" && this.donne.DeviceOrigine=="Francs"){
       this.donne.montant_a_recevoir=this.donne.montant_a_recevoir/655
      }
     
       
    }
    Swal.fire(
      'Transaction effectué!',
      'l_Agent vous doit la somme de '+this.donne.montant_a_recevoir+' '+this.donne.DeviceDest,
      'success'
    )
  }
  else{
    Swal.fire({
      icon: 'error',
      title: 'Oops...',
      text: 'Veuillez vérifier vos information de paiement ',
      footer: '<a href="/">Accueil</a>'
    })
    console.log('suis la');
    
  }
})


  }

}
