import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Component } from '@angular/core';
import { NgForm } from '@angular/forms';
import Swal from 'sweetalert2';
import { AllServicesService } from '../all-services.service';

@Component({
  selector: 'app-creer-agence',
  templateUrl: './creer-agence.component.html',
  styleUrls: ['./creer-agence.component.css']
})
export class CreerAgenceComponent {
  constructor(private http:HttpClient,private serviceagence:AllServicesService){

  }
  httpOptions = {
    headers: new HttpHeaders({
      "Content-Type": "application/json"
    })
  };


  submit (form: NgForm) {
    //username password idSousAgence
    let codeAgence=form.value.codeAgence
    let nomAgence=form.value.nomAgence


   this.serviceagence.InsertAgence(codeAgence,nomAgence,"Actif")

    .subscribe((res) =>{
      if(res.erreur==false){
        Swal.fire(
          'Inscription Passé!',
          'Avec succés',
          'success'
        )
      }


  }
    )
  }

}
